import os
import sys
import logging
from pathlib import Path
from typing import List, Dict
from datetime import datetime
import json
import re
import numpy as np
from numpy.linalg import norm

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
from backend.utils.embeddings import get_embedding

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
MODEL_NAME = "gpt-4o" # Using a more powerful model for this critical reasoning step.
NUM_SNIPPETS_TO_KEEP = 4 # Number of snippets that the reranking process keeps at max
MAX_CANDIDATES_FOR_LLM = 20 # Set a hard limit on the number of candidates sent to the LLM
ALPHA = 0.4 # Weight for blending specificity and similarity scores

# --- Pydantic Model ---
# Defines the expected output structure for the LLM call.
# The LLM will return the integer indices of the snippets it selects.
class RerankedIndices(BaseModel):
    """A list of the 1-based integer indices of the most relevant snippets."""
    reranked_indices: List[int] = Field(
        description="A list of the 1-based integer indices of the top snippets, ordered by relevance."
    )

# --- Prompt Template ---
# This prompt instructs the LLM to act as a data analyst and use engineered features
# to make a more analytical ranking decision.
PROMPT_TEMPLATE = """You are an expert ranking assistant. Your task is to rank a list of candidate documents that may explain a concept drift. Each candidate has been pre-scored with a `priority_score`.

You will receive:

1. A detected concept drift:
   - Drift Type: {drift_type}
   - Drift Phrase: "{drift_phrase}"

2. A list of candidate snippets, each with:
   - index (1-based)
   - document identifier
   - priority_score
   - similarity_score
   - snippet text

Format:
{formatted_snippets}

Your task is to sort all candidates by relevance using these pairwise rules:

1. **Priority**  
   - If |priority_score_A - priority_score_B| > 0.025, the higher priority_score wins.

2. **Similarity**  
   - Otherwise (|Δpriority| ≤ 0.025):  
     - If |similarity_score_A - similarity_score_B| > 0.01, the higher similarity_score wins.

3. **Semantic Judgment**  
   - Otherwise (both differences within thresholds):  
     - Use your semantic understanding of the snippet texts to decide which is more relevant.

**Output**  
Return **only** a JSON object with a single key `"reranked_indices"`, whose value is the list of the top {num_to_keep} 1-based indices, in descending order of relevance.  
Example:  
 ```json
{{ "reranked_indices": [3, 1, 4, 2] }}
"""

def format_snippets_for_reranking(snippets: List[ContextSnippet], start_date: datetime) -> str:
    """Formats snippets into a compact, YAML-like block for the LLM prompt."""
    formatted_str = ""
    # Use 1-based indexing for the candidate labels shown to the LLM
    for i, snippet in enumerate(snippets, 1):
        delta_days = "N/A"
        if snippet.get("timestamp", 0):
            delta = abs((datetime.fromtimestamp(snippet['timestamp']) - start_date).days)
            delta_days = f"{delta} days"
            
        sim_score = snippet.get('score', 0.0)
        sem_spec = snippet.get('semantic_specificity', 0.0)
        
        # Sanitize the snippet text to prevent breaking the prompt format
        text = snippet["snippet_text"].replace('\n', ' ').replace('"', '\\"')

        # Present all features clearly to the LLM for each snippet.
        formatted_str += (
            f"Candidate {i}:\n"
            f"  doc_id: \"{snippet['source_document']}\"\n"
            f"  priority_score: {snippet.get('priority_score', 0.0):.3f}\n"
            f"  similarity_score: {snippet.get('score', 0.0):.3f}\n"
            f"  semantic_specificity: {snippet.get('semantic_specificity', 0.0):.3f}\n"
            f"  snippet: \"{text}\"\n"
        )
    return formatted_str

def run_reranker_agent(state: GraphState) -> dict:
    """
    Ranks candidates using a blended score and a final LLM review step.
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
    drift_phrase = state.get("drift_phrase", "")
    start_date = datetime.fromisoformat(drift_info["start_timestamp"])

    # Step 1: Calculate semantic specificity and blended priority score
    logging.info("Generating embeddings to calculate semantic specificity...")
    drift_emb = get_embedding(drift_phrase)
    for snip in candidate_snippets:
        snippet_emb = get_embedding(snip["snippet_text"])
        sem_spec = float(np.dot(drift_emb, snippet_emb) / (norm(drift_emb) * norm(snippet_emb)))
        snip["semantic_specificity"] = sem_spec

        similarity = snip.get('score', 0.0)
        # Blend the scores using ALPHA
        priority_score = (ALPHA * sem_spec) + ((1 - ALPHA) * similarity)
        snip['priority_score'] = priority_score

    # Step 2: Deduplicate using the new priority score
    unique_by_doc = {}
    for snip in candidate_snippets:
        doc = snip["source_document"]
        if doc not in unique_by_doc or snip["priority_score"] > unique_by_doc[doc]["priority_score"]:
            unique_by_doc[doc] = snip
    candidate_snippets = list(unique_by_doc.values())
            
    # Step 3: Sort candidates by the new priority score
    sorted_candidates = sorted(candidate_snippets, key=lambda x: x.get('priority_score', 0.0), reverse=True)

    logging.info("Top candidates after pre-sorting (by blended priority score):")
    for i, snip in enumerate(sorted_candidates[:5]):
        logging.info(
            f"  > Pre-Rank #{i+1}: {Path(snip['source_document']).name} "
            f"(Priority: {snip.get('priority_score', 0.0):.3f}, SemSpec: {snip.get('semantic_specificity', 0.0):.3f}, Sim: {snip.get('score', 0.0):.3f})"
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
            drift_phrase=drift_phrase,
            formatted_snippets=format_snippets_for_reranking(candidates_for_llm, start_date),
            num_to_keep=NUM_SNIPPETS_TO_KEEP
        )
        
        cache_key = get_cache_key(prompt, MODEL_NAME)
        
        response_data = llm_cache.get(cache_key)
        if not response_data:
            logging.info("CACHE MISS. Calling API for re-ranking...")
            try:
                response_object = structured_llm.invoke(prompt)
                response_data = response_object.dict()
                llm_cache[cache_key] = response_data
                save_to_cache(llm_cache)
                logging.info("Re-ranking response cached successfully.")
            except Exception as e:
                return {"error": f"Error during re-ranking: {e}"}
        else:
            logging.info("CACHE HIT for re-ranking.")

        # Reconstruct exactly what the LLM picked, in order:
        indices = response_data.get("reranked_indices", [])
        llm_picks = [
            candidates_for_llm[i - 1]
            for i in indices
            if 1 <= i <= len(candidates_for_llm)
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
    gold_doc = state["drift_info"].get("gold_doc", "")
    kept_doc_stems = {Path(s["source_document"]).stem.lower() for s in reranked_list}
    logging.info("[Re-rank] kept=%s  gold_in_keep=%s",
                 [Path(s["source_document"]).name for s in reranked_list],
                 "YES ✅" if Path(gold_doc).stem.lower() in kept_doc_stems else "NO ❌")
    # ------------------------------------------------
    
    # --- Step 5: Assemble final output ---
    #final_evidence_docs = [Path(s['source_document']).name for s in reranked_list]
    #final_support_docs = [Path(s['source_document']).name for s in supporting_context]
    #logging.info(f"Final Evidence List: {final_evidence_docs}")
    #logging.info(f"Final Supporting Context: {final_support_docs}")

    return {
        "supporting_context": supporting_context,
        "reranked_context_snippets": reranked_list
    }