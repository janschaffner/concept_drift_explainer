import os
import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta

# --- Path Correction ---
# This adjusts the path so we can import from the 'backend' folder
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
# -----------------------

from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from backend.state.schema import GraphState, ContextSnippet

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# The index name is now loaded from .env, so the hardcoded constant is removed.
EMBEDDING_MODEL_NAME = "text-embedding-3-small" # This model has a 1536 dimension
# Define Namespace Constants
CONTEXT_NS = "context"
KB_NS      = "bpm-kb"
CONTEXT_TOP_K = 30 # Query all vectors
# Skewed temporal window constants
WINDOW_BEFORE = 14
WINDOW_AFTER  = 3      # days *after* start


def run_context_retrieval_agent(state: GraphState) -> dict:
    """
    Retrieves and merges context snippets from both the 'context' and 'bpm-kb' namespaces.

    Args:
        state: The current graph state, which must contain `drift_info` and `drift_keywords`.

    Returns:
        A dictionary with the `raw_context_snippets` field populated with up to 10 candidate
        snippets, of which up to 8 are from the 'context' namespace and up to 2 are
        from the 'bpm-kb' namespace, sorted by relevance.
    """
    logging.info("--- Running Context Retrieval Agent (Dual Namespace Search) ---")
    
    drift_info = state.get("drift_info")
    # Get the keywords extracted by the Drift Agent
    drift_keywords = state.get("drift_keywords", [])

    if not drift_info:
        error_msg = "Drift info not found in state. Cannot retrieve context."
        logging.error(error_msg)
        return {"error": error_msg}

    # 1. Initialize Pinecone & Embedder
    load_dotenv()
    api_key = os.getenv("PINECONE_API_KEY")
    # --- Load index name from .env ---
    pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
    
    if not all([api_key, pinecone_index_name]):
        error_msg = "PINECONE_API_KEY or PINECONE_INDEX_NAME not found in .env file."
        logging.error(error_msg)
        return {"error": error_msg}

    pc = Pinecone(api_key=api_key)
    index = pc.Index(pinecone_index_name) # Use the variable for the index name
    # --- Use OpenAIEmbeddings ---
    embedder = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)

    # 2. Formulate Semantic Query
    # We create a descriptive sentence and enhance it with specific keywords.
    start_activity, end_activity = drift_info["changepoints"]
    process_name = drift_info.get("process_name", "a business process") # Get the process name
    # The query now includes the process name
    base_query = (
        f"A concept drift of type '{drift_info['drift_type']}' was detected. "
        f"It occurred in the '{process_name}' process involving the activities '{start_activity}' and '{end_activity}'."
    )
    
    # Append the specific keywords to the query if they exist
    if drift_keywords:
        keyword_str = ", ".join(drift_keywords)
        query_text = f"{base_query} Associated keywords include: {keyword_str}."
    else:
        query_text = base_query
    
    logging.info(f"Formulated enhanced query: {query_text}")
    query_vector = embedder.embed_query(query_text)

    # 3. Define Temporal Filter (only for the 'context' namespace)
    # Create a time window around the drift to filter documents by metadata.
    # Window: 14 days before the drift started to 3 days after it ended.
    try:
        start_date = datetime.fromisoformat(drift_info["start_timestamp"])
        # Skewed temporal window logic
        filter_start = start_date - timedelta(days=WINDOW_BEFORE)
        filter_end   = start_date + timedelta(days=WINDOW_AFTER)

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

    # 4. Query Both Namespaces and Merge Results
    all_hits = {}

    # Query 'context' namespace with the time filter
    try:
        logging.info(f"Querying '{CONTEXT_NS}' namespace with top_k={CONTEXT_TOP_K}...")
        context_response = index.query(
            vector=query_vector,
            top_k=CONTEXT_TOP_K, # Bring entire corpus
            filter=temporal_filter,
            namespace=CONTEXT_NS,
            include_metadata=True
        )

        # --- Adaptive temporal window fallback ---
        if not context_response.get('matches'):
            logging.info("Fallback: disabled time filter to find more results.")
            context_response = index.query(
                vector=query_vector, 
                top_k=CONTEXT_TOP_K, 
                namespace=CONTEXT_NS, 
                include_metadata=True
            )

        for match in context_response.get('matches', []):
            text_key = match['metadata']['text']
            if text_key not in all_hits or match['score'] > all_hits[text_key]['score']:
                all_hits[text_key] = {'score': match['score'], 'metadata': match['metadata'], 'source_type': CONTEXT_NS}
    except Exception as e:
        logging.error(f"Error querying '{CONTEXT_NS}' namespace: {e}")

    # Query 'bpm-kb' namespace for top 1 hit
    try:
        logging.info(f"Querying '{KB_NS}' namespace for top 1 hit...")
        kb_response = index.query(
            vector=query_vector,
            top_k=1, # Bring only the single most relevant term
            namespace=KB_NS,
            include_metadata=True
        )
        for match in kb_response.get('matches', []):
            text_key = match['metadata']['text']
            # Mark glossary snippets as support-only ---
            match['metadata']['support_only'] = True
            if text_key not in all_hits or match['score'] > all_hits[text_key]['score']:
                all_hits[text_key] = {'score': match['score'], 'metadata': match['metadata'], 'source_type': KB_NS}
    except Exception as e:
        logging.error(f"Error querying '{KB_NS}' namespace: {e}")

    # 5. Sort the combined list by score and process the final snippets
    sorted_hits = sorted(all_hits.values(), key=lambda x: x['score'], reverse=True)
    
    retrieved_snippets: list[ContextSnippet] = []
    if sorted_hits:
        for hit in sorted_hits:
            metadata = hit['metadata']
            snippet: ContextSnippet = {
                "snippet_text": metadata.get("text", ""),
                "source_document": metadata.get("source", "Unknown"),
                "timestamp": metadata.get("timestamp", 0),
                "score": hit.get('score', 0.0), # Add raw score to each snippet
                "classifications": [], # Initialize as an empty list
                "source_type": hit['source_type'] # Add the source tag
            }
            retrieved_snippets.append(snippet)
        logging.info(f"Successfully retrieved and merged {len(retrieved_snippets)} candidate context snippets.")
        # This provides a clear summary of what was retrieved before re-ranking.
        for i, snip in enumerate(retrieved_snippets[:5]): # Log top 5 candidates
            logging.info(
                f"  > Candidate #{i+1}: {Path(snip['source_document']).name} "
                f"(Score: {snip.get('score', 0.0):.3f}, Source: {snip.get('source_type')})"
            )
    else:
        logging.warning("No context snippets found matching the criteria.")

    return {"raw_context_snippets": retrieved_snippets}