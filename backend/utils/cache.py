import json
import hashlib
import logging
from pathlib import Path

# --- Path Correction ---
# This is needed to ensure the cache file is saved in the correct project location
project_root = Path(__file__).resolve().parents[2]
CACHE_DIR = project_root / "data" / "cache"
CACHE_FILE = CACHE_DIR / "llm_cache.json"

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _ensure_cache_dir_exists():
    """Creates the cache directory if it doesn't exist."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

def load_cache() -> dict:
    """Loads the entire cache from the JSON file."""
    _ensure_cache_dir_exists()
    if not CACHE_FILE.exists():
        return {}
    try:
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {}

def save_to_cache(cache: dict):
    """Saves the entire cache back to the JSON file."""
    _ensure_cache_dir_exists()
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)

def get_cache_key(prompt: str, model_name: str) -> str:
    """Creates a unique hash key for a given prompt and model name."""
    # We include the model name in the key so that switching models
    # doesn't return a cached response from a different model.
    hash_input = f"{model_name}:{prompt}"
    return hashlib.md5(hash_input.encode()).hexdigest()