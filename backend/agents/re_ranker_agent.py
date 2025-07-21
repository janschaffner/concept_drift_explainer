import os
import sys
import logging
from pathlib import Path
from typing import List
from datetime import datetime
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
# Set a hard limit on the number of candidates sent to the LLM
MAX_CANDIDATES_FOR_LLM = 20

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

def format_snippets_for_reranking(snippets: List[ContextSnippet], start_date: datetime) -> str:
    """Formats snippets with all engineered features for the LLM prompt."""
    formatted_str = ""
    for i, snippet in enumerate(snippets):
        delta_days = "N/A"
        if snippet.get("timestamp", 0):
            delta = abs((datetime.fromtimestamp(snippet['timestamp']) - start_date).days)
            delta_days = f"{delta} days"
            
        score = snippet.get('score', 0.0)
        # Use the pre-calculated specificity score
        specificity_score = snippet.get('specificity_score', 0.0)
        
        # Sanitize the snippet text to prevent breaking the prompt format
        text = snippet["snippet_text"].replace('\n', ' ').replace('"', '\\"')

        # Present all features clearly to the LLM for each snippet.
        formatted_str += (
            f"### Snippet {i+1} (Source: {snippet['source_document']})\n"
            f"- Similarity Score: {score:.3f}\n"
            f"- Specificity Score: {specificity_score:.1f}\n"
            f"- Δdays from Drift Start: {delta_days}\n"
            f"Text: \"{text}\"\n\n"
        )
    return formatted_str

def run_reranker_agent(state: GraphState) -> dict:
    """
    Force-keeps the best hit, then uses an LLM to re-rank the remaining snippets,
    and finally splits the results into 'evidence' and 'supporting' context lists.
    """
    logging.info("--- Running Re-Ranker Agent ---")

    raw_snippets = state.get("raw_context_snippets", [])
    
    # Step 0: Segregate the KB hit
    supporting_context = []
    candidate_snippets = []
    for snip in raw_snippets:
        if snip.get("source_type") == "bpm-kb":
            supporting_context.append(snip)
        else:
            candidate_snippets.append(snip)
    
    if not candidate_snippets:
        logging.warning("No candidate evidence snippets to re-rank.")
        return {"reranked_context_snippets": [], "supporting_context": supporting_context}

    drift_info = state.get("drift_info", {})
    specific_entities = state.get("specific_entities", [])
    start_date = datetime.fromisoformat(drift_info["start_timestamp"])

    # Step 1: Calculate specificity score first
    for snip in candidate_snippets:
        snip['specificity_score'] = calculate_specificity_score(
            snip['snippet_text'], specific_entities
        )

    # Step 2: Deduplicate, prioritizing specificity
    logging.info(f"Evidence candidates before dedupe: {len(candidate_snippets)}")
    unique_by_doc = {}
    for snip in candidate_snippets:
        doc = snip["source_document"]
        current = unique_by_doc.get(doc)
        # Prefer the snippet with higher (specificity, then similarity)
        if not current or (
            (snip["specificity_score"], snip["score"])
            > (current["specificity_score"], current["score"])
        ):
            unique_by_doc[doc] = snip
    
    candidate_snippets = list(unique_by_doc.values())
    logging.info(f"Evidence candidates after doc-level dedupe: {len(candidate_snippets)}")
            
    # Step 3: Sort candidates for logging
    # Note: This sorting is for human-readable logs; the LLM gets an unsorted view of features.
    sorted_candidates = sorted(candidate_snippets, key=lambda x: (x.get('specificity_score', 0.0), x.get('score', 0.0)), reverse=True)

    # Update log message to be accurate
    logging.info("Top candidates after pre-sorting (by specificity, then similarity):")
    for i, snip in enumerate(sorted_candidates[:5]):
        logging.info(
            f"  > Pre-Rank #{i+1}: {Path(snip['source_document']).name} "
            f"(Specificity: {snip.get('specificity_score', 0.0):.1f}, Score: {snip.get('score', 0.0):.3f})"
        )
    # Step 4: Let the LLM pick the top snippets
    reranked_list = []
    # Short-circuit is triggered only when there is exactly one candidate.
    if len(candidate_snippets) == 1:
        logging.info("Only one candidate found. Skipping LLM re-ranking and keeping it.")
        reranked_list = candidate_snippets
    elif candidate_snippets:
        # Clamp the list to a maximum size before sending to the LLM
        candidates_for_llm = sorted_candidates[:MAX_CANDIDATES_FOR_LLM]
        logging.info(f"Sending top {len(candidates_for_llm)} candidates to LLM for final ranking.")
        
        load_dotenv()
        if not os.getenv("OPENAI_API_KEY"):
            return {"error": "OPENAI_API_KEY not found."}
            
        llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
        structured_llm = llm.with_structured_output(RerankedIndices)
        
        llm_cache = load_cache()
        
        prompt = PROMPT_TEMPLATE.format(
            drift_type=drift_info["drift_type"],
            specific_entities=", ".join(specific_entities),
            formatted_snippets=format_snippets_for_reranking(candidates_for_llm, start_date),
            num_to_keep=NUM_SNIPPETS_TO_KEEP
        )
        
        cache_key = get_cache_key(prompt, MODEL_NAME)
        
        response_data = {}
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

        # Reconstruct exactly what the LLM picked, in order:
        indices = response_data.get("reranked_indices", [])
        llm_picks = [
            candidate_snippets[i - 1]
            for i in indices
            if 1 <= i <= len(candidate_snippets)
        ]

        # Dedupe by document and clamp to NUM_SNIPPETS_TO_KEEP
        seen_docs = set()
        final_evidence = []
        for snip in llm_picks:
            doc = snip["source_document"]
            if doc not in seen_docs:
                seen_docs.add(doc)
                final_evidence.append(snip)
                if len(final_evidence) == NUM_SNIPPETS_TO_KEEP:
                    break
        
        reranked_list = final_evidence

    logging.info(f"Re-ranking complete. Kept {len(reranked_list)} of {len(candidate_snippets)} snippets.")
    
    # ------------------------------------------------
    # Diagnostic logging
    # This clearly states which documents were kept and if the golden document was among them.
    # Normalize filenames using Path().stem for a more robust comparison
    gold_doc_stem = Path(drift_info.get("gold_doc", "")).stem.lower()
    kept_doc_stems = {Path(s["source_document"]).stem.lower() for s in reranked_list}
    logging.info("[Re-rank] kept=%s  gold_in_keep=%s",
                 [Path(s["source_document"]).name for s in reranked_list],
                 "YES ✅" if gold_doc_stem in kept_doc_stems else "NO ❌")
    # ------------------------------------------------
    
    # --- Step 5: Assemble final output ---
    final_evidence_docs = [Path(s['source_document']).name for s in reranked_list]
    final_support_docs = [Path(s['source_document']).name for s in supporting_context]
    logging.info(f"Final Evidence List: {final_evidence_docs}")
    logging.info(f"Final Supporting Context: {final_support_docs}")

    return {
        "supporting_context": supporting_context,
        "reranked_context_snippets": reranked_list
    }