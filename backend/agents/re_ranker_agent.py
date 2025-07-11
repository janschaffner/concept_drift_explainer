import os
import sys
import logging
from datetime import datetime, timedelta
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

# --- Pydantic Model ---
# The LLM will now return the *indices* of the best snippets, not the full text.
class RerankedIndices(BaseModel):
    """A list of the integer indices of the most relevant snippets."""
    reranked_indices: List[int] = Field(
        description="A list of the integer indices of the top snippets, ordered by relevance."
    )

# --- Prompt Template ---
PROMPT_TEMPLATE = """You are an expert business process analyst. Your task is to identify the most relevant evidence to explain a concept drift.
You will be given information about the drift and a numbered list of candidates.

**## 1. Detected Concept Drift**
- **Drift Type:** {drift_type}
- **Drift Keywords:** {drift_keywords}

**## 2. Candidate Snippets (already sorted by initial similarity)**
{formatted_snippets}

**## 3. Your Task**
Read all the candidate snippets carefully. Identify and return ONLY the integer indices of the top {num_to_keep} snippets that are most likely to be the direct cause of the drift.
Your output must be a valid JSON object containing a single key "reranked_indices" with a list of numbers.
"""

def format_snippets_for_reranking(snippets: List[ContextSnippet], start_date: datetime) -> str:
    """Formats snippets with score and date delta for the prompt."""
    formatted_str = ""
    for i, snippet in enumerate(snippets):
        delta_days = "N/A"
        if snippet.get("timestamp", 0):
            delta = abs((datetime.fromtimestamp(snippet['timestamp']) - start_date).days)
            delta_days = f"{delta} days"
        score = snippet.get('score', 0.0)
        formatted_str += f"### Snippet {i+1} (Source: {snippet['source_document']}, Score: {score:.3f}, Δdays: {delta_days})\n"
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
        return {"reranked_context_snippets": [], "supporting_context": []}

    drift_info = state.get("drift_info", {})
    drift_keywords = state.get("drift_keywords", [])

    # --- Add date bonus to scores ---
    start_date = datetime.fromisoformat(drift_info["start_timestamp"])
    for snip in candidate_snippets:
        ts = snip.get('timestamp', 0)
        score = snip.get('score', 0.0)
        if ts:
            delta = abs((datetime.fromtimestamp(ts) - start_date).days)
            if delta <= 7:
                score += 0.10  # boost close docs
            elif delta >= 30:
                score -= 0.10  # demote distant docs
        snip['score'] = score
            
    # Re-sort candidates after applying bonus
    candidate_snippets = sorted(candidate_snippets, key=lambda x: x.get('score', 0.0), reverse=True)

    # --- NEW: Force-keep the single highest-scoring context snippet ---
    # This acts as a safety net to guarantee the best initial hit is considered.
    best_context_hit = next((s for s in candidate_snippets if s.get("source_type") == "context"), None)
    
    final_reranked_list = []
    if best_context_hit:
        final_reranked_list.append(best_context_hit)

    # The list of candidates for the LLM to rank is now everything *except* our force-kept best hit.
    other_candidates = [s for s in candidate_snippets if s != best_context_hit]
    
    # We only call the LLM if there are other candidates to rank.
    if other_candidates:
        load_dotenv()
        if not os.getenv("OPENAI_API_KEY"):
            return {"error": "OPENAI_API_KEY not found."}
            
        llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
        structured_llm = llm.with_structured_output(RerankedIndices)
        
        llm_cache = load_cache()
        
        # We ask the LLM to choose from the remaining candidates.
        # We need 3 more to reach our budget of 4 (1 force-kept + 3 from LLM).
        num_to_keep_from_llm = NUM_SNIPPETS_TO_KEEP - len(final_reranked_list)

        prompt = PROMPT_TEMPLATE.format(
            drift_type=drift_info.get("drift_type", "N/A"),
            drift_keywords=", ".join(drift_keywords),
            formatted_snippets=format_snippets_for_reranking(other_candidates, start_date),
            num_to_keep=num_to_keep_from_llm
        )
        
        cache_key = get_cache_key(prompt, MODEL_NAME)
        
        if cache_key in llm_cache:
            logging.info("CACHE HIT for re-ranking.")
            response_data = llm_cache[cache_key]
        else:
            logging.info("CACHE MISS. Calling API for re-ranking...")
            try:
                response_object = structured_llm.invoke(prompt)
                response_data = response_object.dict()
                llm_cache[cache_key] = response_data
                save_to_cache(llm_cache)
                logging.info("Re-ranking response cached successfully.")
            except Exception as e:
                logging.error(f"Error during re-ranking: {e}")
                return {"error": str(e)}

        # --- Reconstruct the list of full snippets using the returned indices ---
        indices = response_data.get("reranked_indices", [])
        llm_ranked_snippets = [other_candidates[i - 1] for i in indices if 0 < i <= len(other_candidates)]
        
        # Add the LLM's choices to our final list
        final_reranked_list.extend(llm_ranked_snippets)

    # --- De-duplicate the list to prevent errors ---
    seen = set()
    deduped = []
    for snip in final_reranked_list:
        key = (snip["source_document"], snip["snippet_text"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(snip)
    reranked_list = deduped

    logging.info(f"Re-ranking complete. Kept {len(reranked_list)} of {len(candidate_snippets)} snippets.")
    
    #'''
    # --- Diagnostic logging ---
    gold_doc = state["drift_info"].get("gold_doc", "N/A")
    kept_docs = [s["source_document"] for s in reranked_list]
    logging.info("[Re-rank] kept=%s  gold_in_keep=%s",
                 [Path(d).name for d in kept_docs],
                 "YES" if gold_doc in kept_docs else "NO")
    #'''

    # --- Split glossary vs. real evidence ---
    supporting = [s for s in reranked_list if s.get("source_type") == "bpm-kb"]
    evidence   = [s for s in reranked_list if s.get("source_type") == "context"]

    # cap to at most ONE glossary snippet to keep prompts lean
    supporting = supporting[:1]

    # --- Return both lists to update the state correctly ---
    return {
        "supporting_context": supporting,
        "reranked_context_snippets": evidence
    }