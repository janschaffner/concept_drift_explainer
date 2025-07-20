import os
from dotenv import load_dotenv
from pinecone import Pinecone

# --- Configuration ---
NAMESPACE_TO_CLEAR = "context"

# 1. Load API Key and Index Name from .env
load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")

if not all([api_key, pinecone_index_name]):
    print("ERROR: PINECONE_API_KEY or PINECONE_INDEX_NAME not found in .env file.")
    exit()

# 2. Initialize Pinecone
try:
    pc = Pinecone(api_key=api_key)
    index = pc.Index(pinecone_index_name)
    
    print(f"Attempting to clear all vectors from namespace: '{NAMESPACE_TO_CLEAR}'...")
    
    # 3. Call the delete operation
    index.delete(delete_all=True, namespace=NAMESPACE_TO_CLEAR)
    
    print(f"✅ Successfully cleared namespace '{NAMESPACE_TO_CLEAR}'.")
    
except Exception as e:
    print(f"An error occurred: {e}")