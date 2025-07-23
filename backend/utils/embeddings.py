import os
import numpy as np
from functools import lru_cache
from langchain_openai import OpenAIEmbeddings

# --- Configuration ---
EMBEDDING_MODEL_NAME = "text-embedding-3-small"

# --- Cached Embedding Function ---
@lru_cache(maxsize=128)
def get_embedding(text: str) -> np.ndarray:
    """
    Generates a vector embedding for a given text using a cached OpenAI model.
    The lru_cache decorator ensures that repeated calls with the same text
    do not result in redundant API calls.

    Args:
        text: The input text string to embed.

    Returns:
        A numpy array representing the vector embedding.
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found in environment variables.")
    
    # Initialize the embedder within the function to ensure it's thread-safe
    # and respects the cache, but only if an API key is present.
    embedder = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)
    embedding = embedder.embed_query(text)
    return np.array(embedding, dtype=float)