import os
import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta

# --- Path Correction ---
# This adjusts the path so we can import from the 'backend' folder.
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
# -----------------------

from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from backend.state.schema import GraphState, ContextSnippet

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

EMBEDDING_MODEL_NAME = "text-embedding-3-small" # This model has 1536 dimensions.
# Define Namespace Constants for Pinecone.
CONTEXT_NS = "context"
KB_NS      = "bpm-kb"
CONTEXT_TOP_K = 30 # Retrieve a large number of candidates to ensure that the gold doc isn't missed.
# Skewed temporal window constants.
WINDOW_BEFORE = 14
WINDOW_AFTER  = 3 # days *after* start

def run_context_retrieval_agent(state: GraphState, index: Pinecone.Index) -> dict:
    """
    Retrieves and merges context snippets from both the 'context' and 'bpm-kb' namespaces.

    This agent performs a hybrid search. It uses a semantic query enhanced with
    keywords to find relevant documents, while also applying a temporal filter
    to narrow the search space. It queries two separate Pinecone namespaces:
    one for general context documents and one for a BPM-specific knowledge base.

    Args:
        state: The current graph state, which must contain `drift_info` and `drift_keywords`.
        index: An initialized Pinecone index object.
        
    Returns:
        A dictionary with the `raw_context_snippets` field populated with a merged and
        sorted list of candidate snippets for the Re-Ranker Agent.
    """
    logging.info("--- Running Context Retrieval Agent ---")
    
    drift_info = state.get("drift_info")
    # Get the keywords extracted by the Drift Agent.
    drift_keywords = state.get("drift_keywords", [])

    if not drift_info:
        error_msg = "Drift info not found in state. Cannot retrieve context."
        logging.error(error_msg)
        return {"error": error_msg}
    
    # Input Validation)
    changepoints = drift_info.get("changepoints", [])
    if not isinstance(changepoints, (list, tuple)) or len(changepoints) != 2:
        error_msg = f"Invalid changepoints format: Expected a pair, but got {changepoints}"
        logging.error(error_msg)
        return {"error": error_msg}
    start_activity, end_activity = changepoints

    # --- 1. Load .env and check API key before doing any OpenAI calls ---
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        error_msg = "OPENAI_API_KEY not found in .env; cannot embed query."
        logging.error(error_msg)
        return {"error": error_msg}

    # --- 2. Initialize Embedder (after key loaded) ---
    embedder = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)

    # --- 3. Formulate Semantic Query ---
    process_name = drift_info.get("process_name", "a business process") # Get the process name
    # The query includes the process name for better context.
    base_query = (
        f"A concept drift of type '{drift_info['drift_type']}' was detected. "
        f"It occurred in the '{process_name}' process involving the activities '{start_activity}' and '{end_activity}'."
    )
    
    if drift_keywords:
        keyword_str = ", ".join(drift_keywords)
        query_text = f"{base_query} Associated keywords include: {keyword_str}."
    else:
        logging.warning("No keywords or entities extracted; using base query only.")
        query_text = f"{base_query} No additional keywords or entities were extracted."
    
    logging.info(f"Formulated enhanced query: {query_text}")
    
    # Embedding Error Handling
    try:
        query_vector = embedder.embed_query(query_text)
    except Exception as e:
        error_msg = f"Failed to generate query embedding: {e}"
        logging.error(error_msg)
        return {"error": error_msg}

    # --- 3. Define Temporal Filter (only for the 'context' namespace) ---
    # Create a time window around the drift to filter documents by metadata.
    # The window is skewed to prioritize documents published shortly before or after the drift start date.
    try:
        start_date = datetime.fromisoformat(drift_info["start_timestamp"])
        filter_start = start_date - timedelta(days=WINDOW_BEFORE)
        filter_end   = start_date + timedelta(days=WINDOW_AFTER)

        # Convert the filter dates to integer Unix timestamps for the query.
        temporal_filter = {
            "timestamp": {
                "$gte": int(filter_start.timestamp()),
                "$lte": int(filter_end.timestamp())
            }
        }
        logging.info(f"Temporal filter window: {filter_start.date()} to {filter_end.date()}")
    except (ValueError, TypeError) as e:
        logging.warning(f"Could not create temporal filter: {e}. Proceeding without it.")
        temporal_filter = {}

    # --- 4. Query Both Namespaces and Merge Results ---
    all_hits = {}

    # Query 'context' namespace with the time filter
    logging.info(f"Querying '{CONTEXT_NS}' namespace with top_k={CONTEXT_TOP_K}...")
    try:
        if temporal_filter:
            context_response = index.query(
                vector=query_vector,
                top_k=CONTEXT_TOP_K,
                filter=temporal_filter,
                namespace=CONTEXT_NS,
                include_metadata=True
            )
            # Fallback if no matches
            if not getattr(context_response, "matches", []):
                logging.info("Fallback: disabled time filter to find more results.")
                context_response = index.query(
                    vector=query_vector,
                    top_k=CONTEXT_TOP_K,
                    namespace=CONTEXT_NS,
                    include_metadata=True
                )
        else:
            context_response = index.query(
                vector=query_vector,
                top_k=CONTEXT_TOP_K,
                namespace=CONTEXT_NS,
                include_metadata=True
            )
    except Exception as e:
        error_msg = f"Error querying '{CONTEXT_NS}' namespace: {e}"
        logging.error(error_msg)
        return {"error": error_msg}

    # Add context hits to the result dictionary, handling potential duplicates.
    for match in context_response.matches:
        src = match.metadata.get('source', 'Unknown')
        txt = match.metadata.get('text', '')
        dedupe_key = (src, txt)
        if dedupe_key not in all_hits or match.score > all_hits[dedupe_key]['score']:
            all_hits[dedupe_key] = {
                'score': match.score,
                'metadata': match.metadata,
                'source_type': CONTEXT_NS
            }

    # Query 'bpm-kb' namespace for the single best glossary term.
    logging.info(f"Querying '{KB_NS}' namespace for top 1 hit...")
    try:
        kb_response = index.query(
            vector=query_vector,
            top_k=1,
            namespace=KB_NS,
            include_metadata=True
        )
        if not getattr(kb_response, "matches", []):
            logging.warning(f"No results returned from the '{KB_NS}' namespace.")
    except Exception as e:
        error_msg = f"Error querying '{KB_NS}' namespace: {e}"
        logging.error(error_msg)
        return {"error": error_msg}

    for match in kb_response.matches:
        src = match.metadata.get('source', 'Unknown')
        txt = match.metadata.get('text', '')
        dedupe_key = (src, txt)
        match.metadata['support_only'] = True
        if dedupe_key not in all_hits or match.score > all_hits[dedupe_key]['score']:
            all_hits[dedupe_key] = {
                'score': match.score,
                'metadata': match.metadata,
                'source_type': KB_NS
            }

    # --- 5. Sort the combined list by score and process the final snippets ---
    sorted_hits = sorted(all_hits.values(), key=lambda x: x['score'], reverse=True)
    
    retrieved_snippets: list[ContextSnippet] = []
    if sorted_hits:
        for hit in sorted_hits:
            metadata = hit['metadata']
            snippet: ContextSnippet = {
                "snippet_text": metadata.get("text", ""),
                "source_document": metadata.get("source", "Unknown"),
                "timestamp": metadata.get("timestamp", 0),
                "score": hit.get('score', 0.0), # Add raw similarity score to each snippet
                "classifications": [], # Initialize as an empty list.
                "source_type": hit['source_type'], # Add the source tag.
                # Explicitly mark if this came from the glossary namespace
                "support_only": metadata.get("support_only", False)
            }
            retrieved_snippets.append(snippet)

        logging.info(f"Successfully retrieved and merged {len(retrieved_snippets)} candidate context snippets.")
        # This provides a clear summary of what was retrieved before re-ranking.
        for i, snip in enumerate(retrieved_snippets[:5]): # Log top 5 candidates
            logging.info(
                f"  > Candidate #{i+1}: {Path(snip['source_document']).name} "
                f"(Score: {snip.get('score', 0.0):.3f}, Source: {snip.get('source_type')})"
            )
        
        # Debugging: log all docs and gold-doc recall check
        all_docs = [Path(s['source_document']).name for s in retrieved_snippets]
        logging.info(f"All retrieved docs: {all_docs}")

        gold_doc = drift_info.get("gold_doc", "")
        if gold_doc:
        # normalize case to avoid mismatches
            found = "✅" if gold_doc.lower() in [d.lower() for d in all_docs] else "❌"
            logging.info(f"Gold doc: {gold_doc}  Recall@all: {found}")

    else:
        logging.warning("No context snippets found matching the criteria.")

    return {"raw_context_snippets": retrieved_snippets}