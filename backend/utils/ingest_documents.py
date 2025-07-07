import os
import sys
from pathlib import Path
from datetime import datetime
import logging
import io

# --- Path Correction ---
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
# -----------------------

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader # keep this even though it may be old
from langchain_huggingface import HuggingFaceEmbeddings
# Imports for multimodal processing
from backend.utils.image_analyzer import analyze_image_content
from pptx import Presentation

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Pinecone setup
PINECONE_INDEX_NAME = "masterthesis"
PINECONE_DIMENSION = 1024

# Embedding model setup - Must match the Pinecone dimension
EMBEDDING_MODEL_NAME = "sentence-transformers/all-roberta-large-v1" # This model has a 1024 dimension

# Data path
DOCUMENTS_PATH = project_root / "data" / "documents"
# Create a temporary cache directory for extracted images
CACHE_DIR = project_root / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_timestamp_from_filename(filename: str) -> int:
    """Parses yyyy-mm-dd from the start of a filename and returns a Unix timestamp."""
    try:
        date_str = filename.split('_')[0]
        dt_obj = datetime.strptime(date_str, '%Y-%m-%d')
        # Return timestamp as an integer (Unix epoch time)
        return int(dt_obj.timestamp())
    except (ValueError, IndexError):
        return None

def process_and_embed(index, text_splitter, embedder, texts_to_embed: list, source_document_name: str, doc_timestamp: int):
    """Helper function to chunk, embed, and upsert a list of texts."""
    if not texts_to_embed:
        return 0
    
    # Use create_documents to handle a list of texts directly
    documents = text_splitter.create_documents(texts_to_embed)
    
    if not documents:
        logging.warning(f"Text splitting resulted in no chunks for {source_document_name}.")
        return 0
        
    texts = [chunk.page_content for chunk in documents]
    vectors = embedder.embed_documents(texts)
    
    vectors_to_upsert = []
    for i, chunk in enumerate(documents):
        # Create a more unique ID to avoid collisions
        vector_id = f"{Path(source_document_name).stem}_{hash(chunk.page_content)}_{i}"
        metadata = {
            "text": chunk.page_content,
            "source": source_document_name,
            "timestamp": doc_timestamp
        }
        vectors_to_upsert.append((vector_id, vectors[i], metadata))
        
    index.upsert(vectors=vectors_to_upsert)
    logging.info(f"  > Ingested {len(vectors_to_upsert)} vectors from {source_document_name}.")
    return len(vectors_to_upsert)

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
    files_to_process = (
        list(DOCUMENTS_PATH.glob("*.pdf")) +
        list(DOCUMENTS_PATH.glob("*.pptx")) +
        list(DOCUMENTS_PATH.glob("*.png")) +
        list(DOCUMENTS_PATH.glob("*.jpg")) +
        list(DOCUMENTS_PATH.glob("*.docx")) +
        list(DOCUMENTS_PATH.glob("*.txt"))
    )

    if not files_to_process:
        logging.warning(f"No supported documents found in {DOCUMENTS_PATH}. Aborting.")
        return

    logging.info(f"Found {len(files_to_process)} documents to process.")
    
    total_vectors_ingested = 0
    for doc_path in files_to_process:
        logging.info(f"Processing document: {doc_path.name}")
        
        doc_timestamp = get_timestamp_from_filename(doc_path.name)
        if not doc_timestamp:
            logging.warning(f"Could not parse timestamp from '{doc_path.name}'. Skipping file.")
            continue
        
        texts_to_embed = []
        file_suffix = doc_path.suffix.lower()

        if file_suffix in [".png", ".jpg", ".jpeg"]:
            description = analyze_image_content(doc_path)
            if "Error" not in description:
                texts_to_embed.append(description)

        elif file_suffix == ".pptx":
            try:
                prs = Presentation(doc_path)
                for i, slide in enumerate(prs.slides):
                    for shape in slide.shapes:
                        if hasattr(shape, "text"):
                            texts_to_embed.append(shape.text)
                    for shape in slide.shapes:
                        if hasattr(shape, "image"):
                            image = shape.image
                            temp_image_path = CACHE_DIR / f"temp_{image.sha1}.{image.ext}"
                            with open(temp_image_path, "wb") as f:
                                f.write(image.blob)
                            
                            image_description = analyze_image_content(temp_image_path)
                            if "Error" not in image_description:
                                full_description = f"Description of an image from slide {i+1} of '{doc_path.name}': {image_description}"
                                texts_to_embed.append(full_description)
                            os.remove(temp_image_path)
            except Exception as e:
                logging.error(f"Error processing PowerPoint file {doc_path.name}: {e}")
                continue
        
        else: # Default for .pdf, .txt, .docx, etc.
            loader = UnstructuredFileLoader(str(doc_path), strategy="fast")
            documents = loader.load()
            texts_to_embed.extend([doc.page_content for doc in documents])

        if texts_to_embed:
            count = process_and_embed(index, text_splitter, embedder, texts_to_embed, doc_path.name, doc_timestamp)
            total_vectors_ingested += count
        else:
            logging.warning(f"No text or valid images could be extracted from '{doc_path.name}'.")

    logging.info(f"\n--- Ingestion Complete ---")
    logging.info(f"Total vectors ingested into index '{PINECONE_INDEX_NAME}': {total_vectors_ingested}")


if __name__ == "__main__":
    main()