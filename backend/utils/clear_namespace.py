"""
A maintenance utility script to clear all vectors from a specific namespace
in the Pinecone vector database.

Purpose:
    This script is intended for development and testing purposes. It provides a
    quick and easy way to reset the 'context' or 'bpm-kb' knowledge base without
    having to delete and recreate the entire Pinecone index.

Usage:
    1.  Ensure the .env file is correctly configured with the Pinecone API key.
    2.  Set the `NAMESPACE_TO_CLEAR` constant below to the target namespace
        (e.g., "context" or "bpm-kb").
    3.  Run the script from your terminal: `python backend/utils/clear_namespace.py`

WARNING:
    This is a destructive operation and will permanently delete all vectors
    in the specified namespace. There is no confirmation prompt.
"""

import os
from dotenv import load_dotenv
from pinecone import Pinecone

# --- Configuration ---
# Set the name of the namespace that should be cleared.
NAMESPACE_TO_CLEAR = "context"

# Step 1: Load API Key and Index Name from the .env file.
load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")

if not all([api_key, pinecone_index_name]):
    print("ERROR: PINECONE_API_KEY or PINECONE_INDEX_NAME not found in .env file.")
    exit()

# Step 2: Initialize the Pinecone client and connect to the index.
try:
    pc = Pinecone(api_key=api_key)
    index = pc.Index(pinecone_index_name)
    
    print(f"Attempting to clear all vectors from namespace: '{NAMESPACE_TO_CLEAR}'...")
    
    # Step 3: Call the delete operation on the specified namespace.
    index.delete(delete_all=True, namespace=NAMESPACE_TO_CLEAR)
    
    print(f"✅ Successfully cleared namespace '{NAMESPACE_TO_CLEAR}'.")
    
except Exception as e:
    # Catch and report any exceptions during the process.
    print(f"An error occurred: {e}")