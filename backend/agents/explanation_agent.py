import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict
from datetime import datetime

# --- Path Correction ---
# Ensures that the script can correctly import modules from the 'backend' directory.
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
# -----------------------

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pinecone import Pinecone
from pydantic.v1 import BaseModel, Field

from backend.state.schema import GraphState, Explanation
from backend.utils.cache import load_cache, save_to_cache, get_cache_key

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
MODEL_NAME = "gpt-4o-mini"

# --- Pydantic Models for Structured Output ---
# This model defines the expected JSON structure for the LLM's output.
class Cause(BaseModel):
    """Defines the data structure for a single, potential cause of a drift."""
    cause_description: str = Field(description="A cautious, hypothetical analysis of how the evidence could explain the concept drift, citing the source. Frame this as a hypothesis, not a definitive conclusion.")
    evidence_snippet: str = Field(description="The specific text snippet that supports the analysis.")
    source_document: str = Field(description="The name of the source document for the evidence.")
    context_category: str = Field(description="The most relevant Franzoi context category path.")
 
# --- DRIFT PROMPTS: Specialized Prompt Templates for 4 Drift Types ---
# These prompts guide the LLM to generate explanations tailored to the specific
# characteristics of the detected concept drift.

SUDDEN_DRIFT_PROMPT = """You are an expert business process analyst. Your goal is to explain a **Sudden Drift**.
A Sudden Drift involves an abrupt substitution of one process with another. From one point onward, the old process no longer occurs, and all new instances follow the updated version. This type of drift is often triggered by crises, emergencies, or immediate regulatory changes (Bose et al., 2011).
Prioritize evidence that points to a single, discrete event with a specific date.

**## 1. Detected Concept Drift**
- **Drift Type:** {drift_type}
- **Drift Period:** {start_timestamp} to {end_timestamp}

**## 2. Reference Glossary (for your eyes only — DO NOT cite)**
{formatted_glossary}

**## 3. Evidence from Context Documents**
{formatted_evidence}

**## 4. Your Task**
Based on the single block of evidence provided, generate one potential cause for the **SUDDEN** drift. The wording of your explanation must be cautious and hypothetical, not definitive. Use phrases like 'This could suggest...', 'A possible explanation is...', or 'The evidence may indicate...'. Your output must be a single, valid JSON object matching the requested schema.
"""

GRADUAL_DRIFT_PROMPT = """You are an expert business process analyst. Your goal is to explain a **Gradual Drift**.
A Gradual Drift describes a transition phase where both the old and new process variants coexist. This is common in rollout scenarios, where a new process is adopted for new cases, while ongoing cases continue under the old variant. Over time, the older version is phased out entirely (Bose et al., 2011).
Prioritize evidence suggesting a transition, coexistence of old/new processes, or phased rollouts.

**## 1. Detected Concept Drift**
- **Drift Type:** {drift_type}
- **Drift Period:** {start_timestamp} to {end_timestamp}

**## 2. Reference Glossary (for your eyes only — DO NOT cite)**
{formatted_glossary}

**## 3. Evidence from Context Documents**
{formatted_evidence}

**## 4. Your Task**
Based on the single block of evidence provided, generate one potential cause for the **GRADUAL** drift. The wording of your explanation must be cautious and hypothetical, not definitive. Use phrases like 'This could suggest...', 'A possible explanation is...', or 'The evidence may indicate...'. Your output must be a single, valid JSON object matching the requested schema.
 """

INCREMENTAL_DRIFT_PROMPT = """You are an expert business process analyst. Your goal is to explain an **Incremental Drift**.
An Incremental Drift consists of a sequence of small, continuous changes that cumulatively result in significant process transformation. It is often associated with agile BPM practices, where iterative adjustments are made without a single, identifiable change point (Bose et al., 2011; Kraus und van der Aa, 2025).
Prioritize evidence of multiple small adjustments, iterative improvements, or agile practices over time.

**## 1. Detected Concept Drift**
- **Drift Type:** {drift_type}
- **Drift Period:** {start_timestamp} to {end_timestamp}

**## 2. Reference Glossary (for your eyes only — DO NOT cite)**
{formatted_glossary}

**## 3. Evidence from Context Documents**
{formatted_evidence}

**## 4. Your Task**
Based on the single block of evidence provided, generate one potential cause for the **INCREMENTAL** drift. The wording of your explanation must be cautious and hypothetical, not definitive. Use phrases like 'This could suggest...', 'A possible explanation is...', or 'The evidence may indicate...'. Your output must be a single, valid JSON object matching the requested schema.
 """

RECURRING_DRIFT_PROMPT = """You are an expert business process analyst. Your goal is to explain a **Recurring Drift**.
A Recurring Drift occurs when previously observed process versions reappear over time, often in a cyclical pattern. These drifts may follow seasonal cycles or non-periodic triggers (e.g., market-specific promotional workflows) (Bose et al., 2011; Kraus und van der Aa, 2025).
Prioritize evidence of seasonal activities, cyclical patterns, or temporary process changes that are designed to reappear.

**## 1. Detected Concept Drift**
- **Drift Type:** {drift_type}
- **Drift Period:** {start_timestamp} to {end_timestamp}

**## 2. Reference Glossary (for your eyes only — DO NOT cite)**
{formatted_glossary}

**## 3. Evidence from Context Documents**
{formatted_evidence}

**## 4. Your Task**
Based on the single block of evidence provided, generate one potential cause for the **RECURRING** drift. The wording of your explanation must be cautious and hypothetical, not definitive. Use phrases like 'This could suggest...', 'A possible explanation is...', or 'The evidence may indicate...'. Your output must be a single, valid JSON object matching the requested schema.
 """

SUMMARY_PROMPT_TEMPLATE = """You are a senior business process analyst. Based on the following list of potential causes for a concept drift, write a concise, 1-3 sentence executive summary that synthesizes the main findings.

**## Potential Causes**
{formatted_causes}
"""

REFINE_PROMPT_TEMPLATE = """You are a senior editor reviewing an analysis from a junior analyst. 
Your task is to critique and refine the provided "Draft Explanation" based on the original "Evidence" and "Reference Glossary".
Ensure the final summary is concise, the cause descriptions are logical, and that every claim is strongly supported by the cited evidence.

**## Original Reference Glossary (for your eyes only — DO NOT cite)**
{formatted_glossary}

**## Original Evidence**
{formatted_evidence}

**## Draft Explanation to Review**
{draft_explanation}

**## 3. Your Task**
Critique and refine the draft explanation, ensuring the final wording is cautious and hypothetical. Your output must be a single, valid JSON object matching the requested schema.
"""

def format_context_for_prompt(classified_context: list) -> str:
    """Formats the list of classified snippets into a string for the LLM prompt."""
    formatted_str = ""
    for i, snippet in enumerate(classified_context):
        formatted_str += f"### Evidence Snippet {i+1}\n"
        formatted_str += f"- **Source Document:** {snippet['source_document']}\n"
        classifications_str = ", ".join([c['full_path'] for c in snippet.get('classifications', [])])
        formatted_str += f"- **Classified As:** [{classifications_str}]\n"
        formatted_str += f"- **Snippet Text:** \"{snippet['snippet_text']}\"\n\n"
    return formatted_str

def expand_context(snippets: List[Dict], index: Pinecone.Index) -> List[Dict]:
    """For each unique source document, fetches all its chunks from Pinecone.

    This function takes the curated list of snippets, identifies the unique
    source documents, and then queries Pinecone with a metadata filter to
    retrieve the full text of each document. This provides the LLM with the
    complete context rather than just an isolated chunk.

    Args:
        snippets (List[Dict]): The curated list of snippets from the re-ranker.
        index (Pinecone.Index): The initialized Pinecone index connection.

    Returns:
        List[Dict]: A new list of snippet-like objects, where each object
            represents a full document's merged text.
    """
    logging.info("--- Expanding context for final explanation ---")
    # Get a de-duplicated list of source document names to process.
    unique_sources = sorted(list(set(s['source_document'] for s in snippets)))
    
    expanded_docs = []
    for source in unique_sources:
        try:
            # Use a metadata filter to get all chunks for this source document.
            # A dummy vector is passed because we are filtering on metadata only.
            response = index.query(
                vector=[0]*1536, # Dummy vector for metadata-only query.
                filter={"source": source},
                top_k=100, # Assume no single doc has more than 100 chunks.
                namespace="context",
                include_metadata=True
                )
            
            # For debugging, log how many chunks were found for this source.
            logging.debug(f"Found {len(response.matches)} chunks for source '{source}'.")
            
            full_text = " ".join([m.metadata['text'] for m in response.matches])

            # Find the first original snippet that came from this source to copy its metadata.
            original_snippet = next(s for s in snippets if s['source_document'] == source)
            
            # Create a new, expanded document object with the full text.
            expanded_doc = original_snippet.copy()
            expanded_doc["snippet_text"] = full_text
            expanded_docs.append(expanded_doc)
            logging.info(f"  > Expanded '{source}' into a single context block.")
        except Exception as e:
            logging.error(f"Failed to expand context for '{source}': {e}")

    # Return the new list of expanded documents, preserving the original rank order.
    return sorted(expanded_docs, key=lambda x: x.get('score', 0.0), reverse=True)

def calculate_confidence_score(snippet: Dict, drift_info: Dict, rank: int) -> float:
    """Calculates a data-driven confidence score for a given evidence snippet.

    This function uses a multi-stage heuristic model to determine a score.
    It first checks if a snippet meets a minimum semantic similarity threshold.
    If it passes, the snippet starts with a high baseline confidence, to which
    bonuses for specificity (entity matches) and temporal proximity are added.
    The score is then capped at 100% and a final penalty is applied based on
    the snippet's rank in the final list.

    Args:
        snippet (Dict): The evidence snippet object, which must contain the
            'score' (similarity), 'specificity_score', and 'timestamp' keys.
        drift_info (Dict): The dictionary containing information about the drift,
            used here to get the 'start_timestamp'.
        rank (int): The 0-indexed rank of the snippet, used to apply a
            rank-based penalty.

    Returns:
        float: The final, calculated confidence score between 0.0 and 1.0.
    """
    similarity_score = snippet.get("score", 0.0)
    
    # 1. First, check if the document meets a minimum similarity threshold.
    # If not, it's not relevant enough to score highly.
    if similarity_score < 0.20:
        return round(similarity_score * 0.5, 2) # Return a very low score

    # 2. Start with a baseline confidence since it passed the re-ranker and threshold.
    score = 0.85

    # 3. Add a significant bonus for specificity (up to 20%).
    specificity_score = snippet.get("specificity_score", 0.0)
    normalized_specificity = min(1.0, specificity_score / 3.0)
    score += 0.20 * normalized_specificity
 
    # 4. Add a bonus for temporal proximity (up to 15%).
    drift_start_dt = datetime.fromisoformat(drift_info["start_timestamp"])
    evidence_ts = snippet.get("timestamp", 0)
    if evidence_ts > 0:
         evidence_dt = datetime.fromtimestamp(evidence_ts)
         delta_days = abs((drift_start_dt - evidence_dt).days)
    # The bonus decays linearly over 60 days.
    temporal_bonus = 0.15 * max(0.0, 1.0 - (delta_days / 60.0))
    score += temporal_bonus
 
    # 5. Cap the score at 1.0 and apply the final rank penalty.
    capped_score = min(1.0, score)
    rank_bonus = 1.0 if rank == 0 else 0.95 if rank == 1 else 0.90
    final_score = capped_score * rank_bonus

    # Log the detailed breakdown of the score calculation for debugging.
    logging.debug(f"Score breakdown for '{snippet.get('source_document')}':")
    logging.debug(f"  - Baseline: 0.85")
    logging.debug(f"  - Specificity Bonus (score={specificity_score}): +{0.20 * normalized_specificity:.2f}")
    logging.debug(f"  - Temporal Bonus (days_delta={delta_days if evidence_ts > 0 else 'N/A'}): +{temporal_bonus:.2f}")
    logging.debug(f"  - Subtotal (capped): {capped_score:.2f}")
    logging.debug(f"  - Final Score (after rank {rank} penalty of {rank_bonus:.2f}): {final_score:.2f}")

    return round(final_score, 2)

def run_explanation_agent(state: GraphState, index: Pinecone.Index) -> dict:
    """
    Generates a final explanation by individually analyzing each evidence document and then creating a summary.
    """
    logging.info("--- Running Explanation Agent ---")
    
    drift_info = state.get("drift_info")
    # The agent receives the curated evidence from the re-ranker...
    evidence_context = state.get("reranked_context_snippets", [])
    # ...and the supporting glossary terms.
    glossary_context = state.get("supporting_context", [])

    # Log the exact evidence the agent is starting with.
    evidence_sources = [Path(s['source_document']).name for s in evidence_context]
    glossary_sources = [s['source_document'] for s in glossary_context]
    logging.info(f"Agent received {len(evidence_sources)} evidence snippets: {evidence_sources}")
    logging.info(f"Agent received {len(glossary_sources)} support snippets: {glossary_sources}")

    # Filter out any glossary items from the main evidence list.
    # This acts as a guard-rail to ensure glossary items are never used as citable evidence.
    usable_evidence = [
        s for s in evidence_context
        if not s.get("support_only") and s.get("source_type") != "bpm-kb"
    ]

    # Use the Context Expansion feature to fetch the full text for each document.
    usable_evidence = expand_context(usable_evidence, index)

    if not usable_evidence:
        logging.warning("No relevant evidence snippets found after re-ranking. Cannot generate an explanation.")
        no_context_explanation: Explanation = {
            "summary": "No explanation could be generated as no relevant contextual documents were found.",
            "ranked_causes": []
        }
        return {"explanation": no_context_explanation}

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        return {"error": "OPENAI_API_KEY not found."}
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(Cause)
    llm_cache = load_cache()
    cache_updated = False
    final_causes = []
    
    try:
        # --- STEP 1: Loop through each evidence doc and generate a cause ---
        for i, evidence_doc in enumerate(usable_evidence):
            # Format the context for this specific document.
            formatted_glossary = format_context_for_prompt(glossary_context)
            formatted_evidence = format_context_for_prompt([evidence_doc])

            # Dynamically select the prompt based on the type of drift.
            drift_type = drift_info.get('drift_type', '').lower()
            prompt_template = INCREMENTAL_DRIFT_PROMPT
            prompt_name = "INCREMENTAL"
            if 'sudden' in drift_type: 
                prompt_template = SUDDEN_DRIFT_PROMPT
                prompt_name = "SUDDEN"
            elif 'gradual' in drift_type: 
                prompt_template = GRADUAL_DRIFT_PROMPT
                prompt_name = "GRADUAL"
            elif 'recurring' in drift_type: 
                prompt_template = RECURRING_DRIFT_PROMPT
                prompt_name = "RECURRING"
            logging.debug(f"Using {prompt_name}_DRIFT_PROMPT for cause #{i+1}.")

            prompt = prompt_template.format(
                drift_type=drift_info['drift_type'],
                start_timestamp=drift_info['start_timestamp'],
                end_timestamp=drift_info['end_timestamp'],
                formatted_glossary=formatted_glossary,
                formatted_evidence=formatted_evidence
            )
            
            # Check cache before making an expensive API call.
            cache_key = get_cache_key(prompt, MODEL_NAME)
            if cache_key in llm_cache:
                cause_dict = llm_cache[cache_key]
            else:
                response_obj = structured_llm.invoke(prompt)
                cause_dict = response_obj.dict()
                llm_cache[cache_key] = cause_dict
                cache_updated = True
            
            # Calculate the data-driven confidence score and add it to the cause object.
            cause_dict['confidence_score'] = calculate_confidence_score(evidence_doc, drift_info, rank=i)
            final_causes.append(cause_dict)
            logging.info(f"Generated cause for '{evidence_doc['source_document']}' with confidence score {cause_dict['confidence_score']:.2f}")

        # --- STEP 1b: Refine drafts via editor LLM ---
        draft_causes = final_causes.copy()
        refined_causes: list[Dict] = []
        for i, draft in enumerate(draft_causes):
            # prepare the same context pieces
            formatted_glossary = format_context_for_prompt(glossary_context)
            formatted_evidence = format_context_for_prompt([usable_evidence[i]])

            # build and cache the refine prompt
            refine_prompt = REFINE_PROMPT_TEMPLATE.format(
                formatted_glossary=formatted_glossary,
                formatted_evidence=formatted_evidence,
                draft_explanation=json.dumps(draft)
            )
            refine_key = get_cache_key(refine_prompt, MODEL_NAME)
            if refine_key in llm_cache:
                refined = llm_cache[refine_key]
            else:
                logging.info(f"Refining draft cause #{i+1} via LLM…")
                response = structured_llm.invoke(refine_prompt)
                refined = response.dict()
                llm_cache[refine_key] = refined
                cache_updated = True

            # preserve the original confidence_score
            refined['confidence_score'] = draft['confidence_score']
            refined_causes.append(refined)
            logging.info(f"Refined cause #{i+1} for '{draft['source_document']}'")

        # swap in refined causes for the summary
        final_causes = refined_causes

        # --- STEP 2: Generate a summary from all the generated causes ---
        formatted_causes = "\n".join([f"- {c['cause_description']}" for c in final_causes])
        summary_prompt = SUMMARY_PROMPT_TEMPLATE.format(formatted_causes=formatted_causes)
        
        summary_cache_key = get_cache_key(summary_prompt, MODEL_NAME)
        if summary_cache_key in llm_cache:
            summary_text = llm_cache[summary_cache_key]
        else:
            summary_response = llm.invoke(summary_prompt)
            summary_text = summary_response.content
            llm_cache[summary_cache_key] = summary_text
            cache_updated = True
        logging.info(f"  > Generated Summary: {summary_text}")
  
        final_explanation: Explanation = {
            "summary": summary_text,
            "ranked_causes": final_causes
        }
        
        if cache_updated:
            save_to_cache(llm_cache)
            logging.info("LLM cache file updated.")
            
        return {"explanation": final_explanation}

    except Exception as e:
        logging.error(f"Failed to generate final explanation: {e}", exc_info=True)
        return {"error": f"Failed to generate final explanation: {e}"}