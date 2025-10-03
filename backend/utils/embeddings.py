"""
This module provides a centralized and optimized function for generating vector
embeddings using OpenAI's models.

It is designed to be a single source of truth for all embedding generation
across the application. The core function, `get_embedding`, uses an in-memory
LRU (Least Recently Used) cache to prevent redundant API calls for the same
text within a single agent's execution, which improves performance and reduces
costs.
"""

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
    Generates a vector embedding for a given text using a cached model call.

    This function is decorated with @lru_cache, which creates an in-memory
    cache. This ensures that repeated calls with the same text string within a
    single application run do not result in redundant and expensive API calls to
    the OpenAI embedding endpoint.

    Args:
        text: The input text string to embed.

    Returns:
        A numpy array representing the vector embedding.

    Raises:
        ValueError: If the OPENAI_API_KEY environment variable is not set.
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found in environment variables.")
    
    # Initialize the embedder within the function to ensure it's thread-safe
    # and respects the cache, but only if an API key is present.
    embedder = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)
    embedding = embedder.embed_query(text)
    return np.array(embedding, dtype=float)