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
from langchain_community.embeddings import HuggingFaceEmbeddings

# Import our graph state schema
from backend.state.schema import GraphState, ContextSnippet

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PINECONE_INDEX_NAME = "masterthesis"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-roberta-large-v1"
TOP_K_RESULTS = 5 # Retrieve the top 5 most relevant snippets

def run_context_retrieval_agent(state: GraphState) -> dict:
    """
    Retrieves context snippets from Pinecone based on the drift information.

    Args:
        state: The current graph state, which must contain `drift_info`.

    Returns:
        A dictionary with the `raw_context_snippets` field populated.
    """
    logging.info("--- Running Context Retrieval Agent ---")
    
    drift_info = state.get("drift_info")
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
    # We create a descriptive sentence to embed for the semantic search.
    start_activity, end_activity = drift_info["changepoints"]
    query_text = (
        f"A concept drift of type '{drift_info['drift_type']}' was detected. "
        f"It occurred in the process involving the activities '{start_activity}' and '{end_activity}'."
    )
    logging.info(f"Formulated semantic query: {query_text}")
    query_vector = embedder.embed_query(query_text)

    # 3. Define Temporal Filter
    # Create a time window around the drift to filter documents by metadata.
    # Window: 90 days before the drift started to 30 days after it ended.
    try:
        start_date = datetime.fromisoformat(drift_info["start_timestamp"])
        end_date = datetime.fromisoformat(drift_info["end_timestamp"])

        filter_start = start_date - timedelta(days=150)
        filter_end = end_date + timedelta(days=30)

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
            filter=temporal_filter,   # Disable for debugging
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
            text = metadata.get("text", "")
            source = metadata.get("source", "Unknown")
            timestamp = metadata.get("timestamp", "Unknown")

            snippet: ContextSnippet = {
                "snippet_text": text,
                "source_document": source,
                "timestamp": timestamp,
                "franzoi_category": None # To be filled by a later agent
            }
            retrieved_snippets.append(snippet)
        logging.info(f"Successfully retrieved {len(retrieved_snippets)} context snippets.")
    else:
        logging.warning("No context snippets found matching the criteria.")

    return {"raw_context_snippets": retrieved_snippets}