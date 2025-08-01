import json
import hashlib
import logging
from pathlib import Path

# --- Path Correction ---
# This ensures that the cache file is always located relative to the project root,
# making the script runnable from any location.
project_root = Path(__file__).resolve().parents[2]
CACHE_DIR = project_root / "data" / "cache"
CACHE_FILE = CACHE_DIR / "llm_cache.json"

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _ensure_cache_dir_exists():
    """
    Private helper function to create the cache directory if it doesn't already exist.
    This prevents errors when trying to write the cache file for the first time.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

def load_cache() -> dict:
    """
    Loads the entire LLM call cache from the `llm_cache.json` file.

    If the file or directory does not exist, it returns an empty dictionary,
    allowing the system to proceed as if there is no cache.

    Returns:
        A dictionary representing the loaded cache.
    """
    _ensure_cache_dir_exists()
    if not CACHE_FILE.exists():
        return {}
    try:
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        # If the file is corrupted or empty, start fresh --> new cache file.
        return {}

def save_to_cache(cache: dict):
    """
    Saves the provided cache dictionary back to the `llm_cache.json` file.
    
    This function overwrites the existing file with the updated cache,
    persisting the results of new LLM calls for future runs.

    Args:
        cache: The cache dictionary to be saved.
    """
    _ensure_cache_dir_exists()
    with open(CACHE_FILE, "w") as f:
        # Use an indent for readability of the JSON file.
        json.dump(cache, f, indent=2)

def get_cache_key(prompt: str, model_name: str) -> str:
    """
    Creates a unique and deterministic cache key from a prompt and model name.

    The function concatenates the model name and the prompt text and then
    computes an MD5 hash. This ensures that the same prompt sent to the same
    model will always produce the same key, but changing either will result
    in a new key, preventing cache collisions.

    Args:
        prompt: The full text of the prompt sent to the LLM.
        model_name: The name of the LLM being used.

    Returns:
        A string representing the MD5 hash cache key.
    """
    # The model name is included in the key so that switching models
    # doesn't return a cached response from a different model.
    hash_input = f"{model_name}:{prompt}"
    return hashlib.md5(hash_input.encode()).hexdigest()

def clear_llm_cache():
    """
    Deletes the LLM cache file if it exists and returns a status message.
    """
    if CACHE_FILE.exists():
        try:
            CACHE_FILE.unlink()  # This is the modern way to delete a file
            return "✅ LLM cache cleared successfully."
        except Exception as e:
            return f"Error clearing cache: {e}"
    else:
        return "ℹ️ Cache file not found. Nothing to clear."