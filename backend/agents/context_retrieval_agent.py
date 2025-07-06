# backend/agents/context_retrieval_agent.py

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
from langchain_huggingface import HuggingFaceEmbeddings

# Import our graph state schema
from backend.state.schema import GraphState, ContextSnippet

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PINECONE_INDEX_NAME = "masterthesis"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-roberta-large-v1"
# UPDATED: We now retrieve more candidates for the Re-Ranker agent
TOP_K_RESULTS = 15

def run_context_retrieval_agent(state: GraphState) -> dict:
    """
    Retrieves a broad set of context snippets from Pinecone using an enhanced query.

    Args:
        state: The current graph state, which must contain `drift_info` and `drift_keywords`.

    Returns:
        A dictionary with the `raw_context_snippets` field populated with candidate snippets.
    """
    logging.info("--- Running Context Retrieval Agent (Broad Search) ---")
    
    drift_info = state.get("drift_info")
    # NEW: Get the keywords extracted by the Drift Agent
    drift_keywords = state.get("drift_keywords", [])

    if not drift_info:
        error_msg = "Drift info not found in state. Cannot retrieve context."
        logging.error(error_msg)
        return {"error": error_msg}

    # 1. Initialize Pinecone & Embedder
    load_dotenv()
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        error_msg = "PINECONE_API_KEY not found in .env file."
        logging.error(error_msg)
        return {"error": error_msg}

    pc = Pinecone(api_key=api_key)
    index = pc.Index(PINECONE_INDEX_NAME)
    embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # 2. Formulate Semantic Query
    # We create a descriptive sentence and enhance it with specific keywords.
    start_activity, end_activity = drift_info["changepoints"]
    base_query = (
        f"A concept drift of type '{drift_info['drift_type']}' was detected. "
        f"It occurred in the process involving the activities '{start_activity}' and '{end_activity}'."
    )
    
    # NEW: Append the specific keywords to the query if they exist
    if drift_keywords:
        keyword_str = ", ".join(drift_keywords)
        query_text = f"{base_query} Associated keywords include: {keyword_str}."
    else:
        query_text = base_query
    
    logging.info(f"Formulated enhanced query: {query_text}")
    query_vector = embedder.embed_query(query_text)

    # 3. Define Temporal Filter
    # Create a time window around the drift to filter documents by metadata.
    # Window: 14 days before the drift started to 14 days after it ended.
    try:
        start_date = datetime.fromisoformat(drift_info["start_timestamp"])
        end_date = datetime.fromisoformat(drift_info["end_timestamp"])

        filter_start = start_date - timedelta(days=14)
        filter_end = end_date + timedelta(days=14)

        # Convert the filter dates to integer Unix timestamps for the query
        temporal_filter = {
            "timestamp": {
                "$gte": int(filter_start.timestamp()),
                "$lte": int(filter_end.timestamp())
            }
        }
        logging.info(f"Temporal filter window: {filter_start.date()} to {filter_end.date()}")
    except (ValueError, TypeError) as e:
        error_msg = f"Could not create temporal filter due to invalid timestamps: {e}"
        logging.error(error_msg)
        return {"error": error_msg}

    # 4. Query Pinecone
    logging.info(f"Querying Pinecone index '{PINECONE_INDEX_NAME}' with top_k={TOP_K_RESULTS}...")
    try:
        query_response = index.query(
            vector=query_vector,
            filter=temporal_filter,
            top_k=TOP_K_RESULTS,
            include_metadata=True
        )
    except Exception as e:
        error_msg = f"An error occurred while querying Pinecone: {e}"
        logging.error(error_msg)
        return {"error": error_msg}
        
    # 5. Process Results and Update State
    retrieved_snippets: list[ContextSnippet] = []
    if query_response.get("matches"):
        for match in query_response["matches"]:
            metadata = match.get("metadata", {})
            # UPDATED: The data structure now uses 'classifications' as per our latest schema
            snippet: ContextSnippet = {
                "snippet_text": metadata.get("text", ""),
                "source_document": metadata.get("source", "Unknown"),
                "timestamp": metadata.get("timestamp", 0),
                "classifications": [] # Initialize as an empty list
            }
            retrieved_snippets.append(snippet)
        logging.info(f"Successfully retrieved {len(retrieved_snippets)} candidate context snippets.")
    else:
        logging.warning("No context snippets found matching the criteria.")

    return {"raw_context_snippets": retrieved_snippets}