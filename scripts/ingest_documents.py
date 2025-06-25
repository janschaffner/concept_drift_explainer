import os
import sys
from pathlib import Path
from datetime import datetime
import logging

# --- Path Correction ---
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
# -----------------------

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Pinecone setup
PINECONE_INDEX_NAME = "masterthesis"
PINECONE_DIMENSION = 1024

# Embedding model setup - Must match the Pinecone dimension
EMBEDDING_MODEL_NAME = "sentence-transformers/all-roberta-large-v1" # This model has a 1024 dimension

# Data path
DOCUMENTS_PATH = project_root / "data" / "documents"

def get_timestamp_from_filename(filename: str) -> int:
    """Parses YYYY-MM-DD from the start of a filename and returns a Unix timestamp."""
    try:
        date_str = filename.split('_')[0]
        dt_obj = datetime.strptime(date_str, '%Y-%m-%d')
        # Return timestamp as an integer (Unix epoch time)
        return int(dt_obj.timestamp())
    except (ValueError, IndexError):
        return None

def main():
    """
    Main function to load, chunk, embed, and ingest documents into Pinecone.
    """
    logging.info("--- Starting Document Ingestion ---")
    
    # 1. Load API Key
    load_dotenv()
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        logging.error("PINECONE_API_KEY not found in .env file.")
        return

    # 2. Initialize Pinecone
    pc = Pinecone(api_key=api_key)
    
    # 3. Check for and create the index if it doesn't exist
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        logging.info(f"Index '{PINECONE_INDEX_NAME}' not found. Creating a new one...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=PINECONE_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1") # A default serverless spec
        )
        logging.info(f"Index '{PINECONE_INDEX_NAME}' created successfully.")
    else:
        logging.info(f"Connected to existing index: '{PINECONE_INDEX_NAME}'")

    index = pc.Index(PINECONE_INDEX_NAME)

    # 4. Initialize Embedder and Text Splitter
    logging.info(f"Loading embedding model: '{EMBEDDING_MODEL_NAME}'...")
    embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    
    # 5. Process and Ingest Documents
    files_to_process = list(DOCUMENTS_PATH.glob("*.*"))
    if not files_to_process:
        logging.warning(f"No documents found in {DOCUMENTS_PATH}. Aborting.")
        return

    logging.info(f"Found {len(files_to_process)} documents to process.")
    
    total_vectors_ingested = 0
    for doc_path in files_to_process:
        logging.info(f"Processing document: {doc_path.name}")
        
        # Parse timestamp from filename
        doc_timestamp = get_timestamp_from_filename(doc_path.name)
        if not doc_timestamp:
            logging.warning(f"Could not parse timestamp from '{doc_path.name}'. Skipping file.")
            continue
            
        # Load and split the document
        loader = UnstructuredFileLoader(str(doc_path), strategy="fast")
        documents = loader.load()
        chunks = text_splitter.split_documents(documents)
        
        if not chunks:
            logging.warning(f"No text could be extracted from '{doc_path.name}'. Skipping.")
            continue
            
        # Embed text chunks
        texts = [chunk.page_content for chunk in chunks]
        vectors = embedder.embed_documents(texts)
        
        # Prepare vectors for Pinecone upsert
        vectors_to_upsert = []
        for i, chunk in enumerate(chunks):
            vector_id = f"{doc_path.stem}_{i}"
            metadata = {
                "text": chunk.page_content,
                "source": doc_path.name,
                "timestamp": doc_timestamp
            }
            vectors_to_upsert.append((vector_id, vectors[i], metadata))
            
        # Upsert to Pinecone
        index.upsert(vectors=vectors_to_upsert)
        total_vectors_ingested += len(vectors_to_upsert)
        logging.info(f"  > Ingested {len(vectors_to_upsert)} vectors for this document.")

    logging.info(f"\n--- Ingestion Complete ---")
    logging.info(f"Total vectors ingested into index '{PINECONE_INDEX_NAME}': {total_vectors_ingested}")


if __name__ == "__main__":
    main()