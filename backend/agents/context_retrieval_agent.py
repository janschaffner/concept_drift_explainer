"""
This module contains the implementation of the Context Retrieval Agent.

This agent is the first stage in the two-stage retrieval process. Its primary
responsibility is to perform a broad, hybrid search to gather a wide set of
potentially relevant documents ("candidate snippets") from the Pinecone vector
database. It combines semantic search with temporal filtering and queries both a
general context knowledge base and a specialized BPM glossary.
"""

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
# Define constants for the skewed temporal window.
WINDOW_BEFORE = 14 # days *before* start
WINDOW_AFTER  = 3 # days *after* start

def run_context_retrieval_agent(state: GraphState, index: Pinecone.Index) -> dict:
    """Performs a hybrid search to retrieve candidate context snippets.

    This agent acts as the "collector" in the pipeline. It formulates a rich
    semantic query using the drift information and keywords from the Drift Agent.
    It then queries two separate Pinecone namespaces:
    1.  'context': A broad search for general context documents, filtered by a
        skewed temporal window around the drift's start date.
    2.  'bpm-kb': A targeted search for the single most relevant term from a
        curated BPM knowledge base.

    The results are then merged, deduplicated, and sorted by similarity score
    before being passed to the Re-Ranker Agent.

    Args:
        state: The current graph state, which must contain `drift_info`.
        index: An initialized Pinecone index object (dependency injected).

    Returns:
        A dictionary with the `raw_context_snippets` field populated with a
        merged and sorted list of candidate snippets.
    """
    logging.info("--- Running Context Retrieval Agent ---")
    
    # Step 1: Extract necessary data from the state and perform input validation.
    drift_info = state.get("drift_info")
    # Get the keywords extracted by the Drift Agent.
    drift_keywords = state.get("drift_keywords", [])

    if not drift_info:
        error_msg = "Drift info not found in state. Cannot retrieve context."
        logging.error(error_msg)
        return {"error": error_msg}
    
    # Input Validation for changepoints.
    changepoints = drift_info.get("changepoints", [])
    if not isinstance(changepoints, (list, tuple)) or len(changepoints) != 2:
        error_msg = f"Invalid changepoints format: Expected a pair, but got {changepoints}"
        logging.error(error_msg)
        return {"error": error_msg}
    start_activity, end_activity = changepoints

    # Step 2: Initialize the embedding model.
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        error_msg = "OPENAI_API_KEY not found in .env; cannot embed query."
        logging.error(error_msg)
        return {"error": error_msg}
    embedder = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)

    # Step 3: Formulate the rich semantic query by combining the drift type,
    # process name, activities, and extracted keywords.
    process_name = drift_info.get("process_name", "a business process") # Get the process name
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

    # Step 4: Define a skewed temporal filter to prioritize documents published
    # shortly before or after the drift's start date.
    try:
        start_date = datetime.fromisoformat(drift_info["start_timestamp"])
        filter_start = start_date - timedelta(days=WINDOW_BEFORE)
        filter_end   = start_date + timedelta(days=WINDOW_AFTER)

        # Convert dates to integer Unix timestamps for the Pinecone query.
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

    # Step 5: Query both the 'context' and 'bpm-kb' namespaces in Pinecone.
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
            # Fallback mechanism: If the narrow temporal filter returns no results,
            # re-query without the filter to ensure some context is always found.
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

    # Use a dictionary with a (source, text) tuple as the key to automatically
    # handle deduplication of snippets, keeping only the highest-scoring version.
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
        # Explicitly flag glossary items so they can be handled separately.
        match.metadata['support_only'] = True
        if dedupe_key not in all_hits or match.score > all_hits[dedupe_key]['score']:
            all_hits[dedupe_key] = {
                'score': match.score,
                'metadata': match.metadata,
                'source_type': KB_NS
            }

    # Step 6: Merge, sort, and format the final list of candidate snippets.
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

        # Diagnostic check to see if the gold document was retrieved (in evaluation).
        gold_doc = drift_info.get("gold_doc", "")
        if gold_doc:
        # Normalize case to avoid mismatches
            found = "✅" if gold_doc.lower() in [d.lower() for d in all_docs] else "❌"
            logging.info(f"Gold doc: {gold_doc}  Recall@all: {found}")

    else:
        logging.warning("No context snippets found matching the criteria.")

    return {"raw_context_snippets": retrieved_snippets}