import os
import sys
import logging
from pathlib import Path
from typing import List

# --- Path Correction ---
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
# -----------------------

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic.v1 import BaseModel, Field

from backend.state.schema import GraphState, ContextSnippet
from backend.utils.cache import load_cache, save_to_cache, get_cache_key

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
MODEL_NAME = "gpt-4o-mini"
NUM_SNIPPETS_TO_KEEP = 5 # How many snippets to keep after re-ranking

# --- Pydantic Model for Structured Output ---
class RerankedSnippets(BaseModel):
    """A list of the most relevant snippets that best explain the drift."""
    reranked_snippets: List[ContextSnippet] = Field(
        description="A list of the top snippets, ordered by relevance."
    )

# --- Prompt Template ---
PROMPT_TEMPLATE = """You are an expert business analyst. Your task is to identify the most relevant evidence to explain a concept drift.
You will be given information about the drift and a list of candidate text snippets retrieved from a keyword search.

**## 1. Detected Concept Drift**
- **Drift Type:** {drift_type}
- **Drift Keywords:** {drift_keywords}

**## 2. Candidate Snippets**
{formatted_snippets}

**## 3. Your Task**
Read all the candidate snippets carefully. Identify and return ONLY the top {num_to_keep} snippets that are most likely to be the direct cause of the drift.
It is critical that you return them in the same format as the input. Your output must be a valid JSON object containing a single key "reranked_snippets".
"""

def format_snippets_for_reranking(snippets: List[ContextSnippet]) -> str:
    """Formats a list of snippets into a numbered string for the prompt."""
    formatted_str = ""
    for i, snippet in enumerate(snippets):
        formatted_str += f"### Snippet {i+1} (Source: {snippet['source_document']})\n"
        formatted_str += f"{snippet['snippet_text']}\n\n"
    return formatted_str

def run_reranker_agent(state: GraphState) -> dict:
    """
    Uses an LLM to re-rank the retrieved snippets and keep only the most relevant ones.
    """
    logging.info("--- Running Re-Ranker Agent ---")
    
    candidate_snippets = state.get("raw_context_snippets", [])
    if not candidate_snippets:
        logging.warning("No candidate snippets to re-rank.")
        return {"reranked_context_snippets": []}

    drift_info = state.get("drift_info", {})
    drift_keywords = state.get("drift_keywords", [])

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        return {"error": "OPENAI_API_KEY not found."}
        
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
    structured_llm = llm.with_structured_output(RerankedSnippets)
    
    # --- Caching Logic ---
    llm_cache = load_cache()
    
    prompt = PROMPT_TEMPLATE.format(
        drift_type=drift_info.get("drift_type", "N/A"),
        drift_keywords=", ".join(drift_keywords),
        formatted_snippets=format_snippets_for_reranking(candidate_snippets),
        num_to_keep=NUM_SNIPPETS_TO_KEEP
    )
    
    cache_key = get_cache_key(prompt, MODEL_NAME)
    
    if cache_key in llm_cache:
        logging.info("CACHE HIT for re-ranking.")
        response_data = llm_cache[cache_key]
        reranked_list = response_data.get("reranked_snippets", [])
    else:
        logging.info("CACHE MISS. Calling API for re-ranking...")
        try:
            response_object = structured_llm.invoke(prompt)
            response_data = response_object.dict()
            reranked_list = response_data.get("reranked_snippets", [])
            
            llm_cache[cache_key] = response_data
            save_to_cache(llm_cache)
            logging.info("Re-ranking response cached successfully.")
        except Exception as e:
            logging.error(f"Error during re-ranking: {e}")
            return {"error": str(e)}

    logging.info(f"Re-ranking complete. Kept {len(reranked_list)} of {len(candidate_snippets)} snippets.")
    
    # --- Split glossary vs. real evidence ---
    supporting = [s for s in reranked_list if s.get("source_type") == "bpm-kb"]
    evidence   = [s for s in reranked_list if s.get("source_type") == "context"]

    # cap to at most ONE glossary snippet to keep prompts lean
    supporting = supporting[:1]

    state["supporting_context"]        = supporting       # NEW key
    state["reranked_context_snippets"] = evidence         # overwrite with real docs

    # The key for the new list of snippets
    return {"reranked_context_snippets": reranked_list}