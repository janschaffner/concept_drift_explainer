import os
import sys
import logging
from pathlib import Path
from typing import List, Dict
from datetime import datetime, timedelta
import json
import re

# --- Path Correction ---
# Ensures that the script can correctly import modules from the 'backend' directory.
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
# -----------------------

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic.v1 import BaseModel, Field

# Import graph state schema and caching utility
from backend.state.schema import GraphState, ContextSnippet
from backend.utils.cache import load_cache, save_to_cache, get_cache_key

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
MODEL_NAME = "gpt-4o" # Using a more powerful model for this critical reasoning step.
# Keep 1 best hit automatically (force keep mechanism) and ask the LLM to select 3 more.
NUM_SNIPPETS_TO_KEEP = 4

# --- Pydantic Model ---
# Defines the expected output structure for the LLM call.
# The LLM will return the integer indices of the snippets it selects.
class RerankedIndices(BaseModel):
    """A list of the integer indices of the most relevant snippets."""
    reranked_indices: List[int] = Field(
        description="A list of the integer indices of the top snippets, ordered by relevance."
    )

# --- Prompt Template ---
# This prompt instructs the LLM to act as a data analyst and use engineered features
# to make a more analytical ranking decision.
PROMPT_TEMPLATE = """You are an expert data analyst. Your task is to identify the most relevant evidence to explain a concept drift by analyzing a set of features for each candidate document.

**## 1. Detected Concept Drift**
- **Drift Type:** {drift_type}
- **Specific Entities:** {specific_entities}

**## 2. Candidate Snippets with Features**
{formatted_snippets}

**## 3. Your Task**
Review the candidate snippets and their features (Similarity Score, Specificity Score, Δdays). A high **Specificity Score** is a very strong signal of direct relevance.
Based on the combination of these features, identify and return ONLY the integer indices of the top {num_to_keep} snippets that are most likely to be the direct cause of the drift.
Your output must be a valid JSON object containing a single key "reranked_indices" with a list of numbers.
"""

# Helper function for Specificity Score.
def calculate_specificity_score(text: str, specific_entities: List[str]) -> float:
    """Calculates a score based on the presence of specific, unique entities."""
    if not specific_entities:
        return 0.0
    
    score = 0
    text_lower = text.lower()
    for entity in specific_entities:
        # Use regex for whole word matching to avoid matching substrings.
        if re.search(r'\b' + re.escape(entity.lower()) + r'\b', text_lower):
            score += 1 # Add 1 for each specific entity found.
            
    return float(score)

def format_snippets_for_reranking(snippets: List[ContextSnippet], start_date: datetime, specific_entities: List[str]) -> str:
    """Formats snippets with all engineered features for the LLM prompt."""
    formatted_str = ""
    for i, snippet in enumerate(snippets):
        delta_days = "N/A"
        if snippet.get("timestamp", 0):
            delta = abs((datetime.fromtimestamp(snippet['timestamp']) - start_date).days)
            delta_days = f"{delta} days"
            
        score = snippet.get('score', 0.0)
        specificity_score = calculate_specificity_score(snippet['snippet_text'], specific_entities)
        
        # Present all features clearly to the LLM for each snippet.
        formatted_str += (
            f"### Snippet {i+1} (Source: {snippet['source_document']})\n"
            f"- Similarity Score: {score:.3f}\n"
            f"- Specificity Score: {specificity_score:.1f}\n"
            f"- Δdays from Drift Start: {delta_days}\n"
            f"Text: \"{snippet['snippet_text']}\"\n\n"
        )
    return formatted_str

def run_reranker_agent(state: GraphState) -> dict:
    """
    Force-keeps the best hit, then uses an LLM to re-rank the remaining snippets,
    and finally splits the results into 'evidence' and 'supporting' context lists.
    """
    logging.info("--- Running Re-Ranker Agent ---")
    
    candidate_snippets = state.get("raw_context_snippets", [])
    if not candidate_snippets:
        logging.warning("No candidate snippets to re-rank.")
        return {"reranked_context_snippets": [], "supporting_context": []}

    drift_info = state.get("drift_info", {})
    specific_entities = state.get("specific_entities", [])
    start_date = datetime.fromisoformat(drift_info["start_timestamp"])

    # Add date bonus to scores
    # This heuristic boosts the relevance of documents published close to the drift start date.
    start_date = datetime.fromisoformat(drift_info["start_timestamp"])
    for snip in candidate_snippets:
        ts = snip.get('timestamp', 0)
        score = snip.get('score', 0.0)
        if ts:
            delta = abs((datetime.fromtimestamp(ts) - start_date).days)
            if delta <= 7:
                score += 0.10  # boost close docs (CHANGE TO 10 IF WORSE)
            elif delta >= 30:
                score -= 0.10  # demote distant docs
        snip['score'] = score

    # Calculate and attach specificity score to each snippet (needed for confidence score)
    for snip in candidate_snippets:
        snip['specificity_score'] = calculate_specificity_score(
            snip['snippet_text'], specific_entities
        )
            
    # Re-sort candidates after applying the date-based score bonus.
    candidate_snippets = sorted(candidate_snippets, key=lambda x: x.get('score', 0.0), reverse=True)

    # Log the top candidates and their scores before the LLM re-ranks them.
    logging.info("Top candidates after date bonus and pre-sorting:")
    for i, snip in enumerate(candidate_snippets[:5]): # Log top 5
        logging.info(
            f"  > Pre-Rank #{i+1}: {Path(snip['source_document']).name} "
            f"(Score: {snip.get('score', 0.0):.3f})"
        )

    # Force-keep the single highest-scoring context snippet.
    # This acts as a safety net to guarantee the best initial hit is always considered.
    best_context_hit = next((s for s in candidate_snippets if s.get("source_type") == "context"), None)
    
    reranked_list = []
    if best_context_hit:
        reranked_list.append(best_context_hit)

    ## The list of candidates for the LLM to rank is now everything *except* the force-kept best hit.
    other_candidates = [s for s in candidate_snippets if s != best_context_hit]
    
    # We only call the LLM if there are other candidates left to rank.
    if other_candidates:
        load_dotenv()
        if not os.getenv("OPENAI_API_KEY"):
            return {"error": "OPENAI_API_KEY not found."}
            
        llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
        structured_llm = llm.with_structured_output(RerankedIndices)
        
        llm_cache = load_cache()
        
        # # Ask the LLM to choose N-1 snippets, since there was already one force-kept.
        # 3 more are needed to reach the budget of 4 (1 force-kept + 3 from LLM).
        num_to_keep_from_llm = NUM_SNIPPETS_TO_KEEP - len(reranked_list)

        # The prompt uses the specific entities and the feature-rich formatter.
        prompt = PROMPT_TEMPLATE.format(
            drift_type=drift_info.get("drift_type", "N/A"),
            specific_entities=", ".join(specific_entities),
            formatted_snippets=format_snippets_for_reranking(other_candidates, start_date, specific_entities),
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

        # Reconstruct the list of full snippets using the returned indices.
        indices = response_data.get("reranked_indices", [])
        # The LLM returns 1-based indices, so subtract 1 for 0-based list access.
        llm_ranked_snippets = [other_candidates[i - 1] for i in indices if 0 < i <= len(other_candidates)]
        
        # Add the LLM's choices to our final list.
        reranked_list.extend(llm_ranked_snippets)

    # De-duplicate the list to prevent errors.
    seen = set()
    deduped = []
    for snip in reranked_list:
        key = (snip["source_document"], snip["snippet_text"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(snip)
    reranked_list = deduped

    logging.info(f"Re-ranking complete. Kept {len(reranked_list)} of {len(candidate_snippets)} snippets.")
    
    # Diagnostic logging
    # This clearly states which documents were kept and if the golden document was among them.
    gold_doc = state["drift_info"].get("gold_doc", "").lower()
    kept_docs = [s["source_document"].lower() for s in reranked_list]
    logging.info("[Re-rank] kept=%s  gold_in_keep=%s",
                 [Path(d).name for d in kept_docs],
                 "YES ✅" if gold_doc in kept_docs else "NO ❌")

    # Split glossary vs. real evidence
    supporting = [s for s in reranked_list if s.get("source_type") == "bpm-kb"]
    evidence   = [s for s in reranked_list if s.get("source_type") == "context"]

    # Cap to at most ONE glossary snippet to keep prompts lean
    supporting = supporting[:1]
    
    # Log the final evidence and support lists being passed to the next agent.
    final_evidence_docs = [Path(s['source_document']).name for s in evidence]
    final_support_docs = [Path(s['source_document']).name for s in supporting]
    logging.info(f"Final Evidence List: {final_evidence_docs}")
    logging.info(f"Final Supporting Context: {final_support_docs}")


    # Return both lists to update the state correctly.
    return {
        "supporting_context": supporting,
        "reranked_context_snippets": evidence
    }