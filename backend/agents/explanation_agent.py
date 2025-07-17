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
    cause_description: str = Field(description="A cautious analysis of how the evidence could potentially explain the concept drift. Frame this as a hypothesis, not a definitive conclusion.")
    evidence_snippet: str = Field(description="The specific text snippet that supports the analysis.")
    source_document: str = Field(description="The name of the source document for the evidence.")
    context_category: str = Field(description="The most relevant Franzoi context category path.")
 
class ExplanationOutput(BaseModel):
    """Defines the top-level object the LLM should produce for an explanation."""
    summary: str = Field(description="A 1-3 sentence executive summary of the most likely cause.")
    ranked_causes: List[Cause] = Field(description="A list of potential causes, ordered from most to least likely.")


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
Your task is to hypothesize potential reasons for the **SUDDEN** drift based only on the provided evidence. The wording of your explanation must be cautious and hypothetical, not definitive. Use phrases like 'This could suggest...', 'A possible explanation is...', or 'The evidence may indicate...'. Cite only documents from the Evidence section. Structure your response as a valid JSON object with "summary" and "ranked_causes" keys.
- **"summary"**: A 1-3 sentence executive summary of the most likely cause.
- **"ranked_causes"**: A list of potential causes. **The evidence is already correctly ranked; describe each cause in the order the evidence is provided.**
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
Your task is to hypothesize potential reasons for the **GRADUAL** drift based only on the provided evidence. The wording of your explanation must be cautious and hypothetical, not definitive. Use phrases like 'This could suggest...', 'A possible explanation is...', or 'The evidence may indicate...'. Cite only documents from the Evidence section. Structure your response as a valid JSON object with "summary" and "ranked_causes" keys.
- **"summary"**: A 1-3 sentence executive summary of the most likely cause.
- **"ranked_causes"**: A list of potential causes. **The evidence is already correctly ranked; describe each cause in the order the evidence is provided.**
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
Your task is to hypothesize potential reasons for the **INCREMENTAL** drift based only on the provided evidence. The wording of your explanation must be cautious and hypothetical, not definitive. Use phrases like 'This could suggest...', 'A possible explanation is...', or 'The evidence may indicate...'. Cite only documents from the Evidence section. Structure your response as a valid JSON object with "summary" and "ranked_causes" keys.
- **"summary"**: A 1-3 sentence executive summary of the most likely cause.
- **"ranked_causes"**: A list of potential causes. **The evidence is already correctly ranked; describe each cause in the order the evidence is provided.**
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
Your task is to hypothesize potential reasons for the **RECURRING** drift based only on the provided evidence. The wording of your explanation must be cautious and hypothetical, not definitive. Use phrases like 'This could suggest...', 'A possible explanation is...', or 'The evidence may indicate...'. Cite only documents from the Evidence section. Structure your response as a valid JSON object with "summary" and "ranked_causes" keys.
- **"summary"**: A 1-3 sentence executive summary of the most likely cause.
- **"ranked_causes"**: A list of potential causes. **The evidence is already correctly ranked; describe each cause in the order the evidence is provided.**
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
Critique and refine the draft explanation, ensuring the final wording is cautious and hypothetical, not definitive. Use phrases like 'This could suggest...', 'A possible explanation is...', or 'The evidence may indicate...'. **The causes in the draft are already correctly ranked; describe them in the exact order they are provided.** Cite only documents listed in the Evidence section; do NOT cite glossary items as formal evidence. Your output MUST be a valid JSON object in the same format as the draft, with "summary" and "ranked_causes" keys.
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

def calculate_confidence_score(snippet: Dict, drift_info: Dict, rank: int) -> float:
    """Calculates a data-driven confidence score for a given evidence snippet.

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
    temporal_bonus = 0.15 * max(0.0, 1.0 - (delta_days / 60.0))
    score += temporal_bonus
 
    # 5. Cap the score at 1.0 and apply the final rank penalty.
    capped_score = min(1.0, score)
    rank_bonus = 1.0 if rank == 0 else 0.95 if rank == 1 else 0.90
    final_score = capped_score * rank_bonus
    return round(final_score, 2)


def run_explanation_agent(state: GraphState) -> dict:
    """
    Generates and then calibrates a final explanation using a two-step "draft and refine" chain.
    """
    logging.info("--- Running Explanation Agent ---")
    
    drift_info = state.get("drift_info")
    # The agent receives the curated evidence from the re-ranker...
    evidence_context = state.get("reranked_context_snippets", [])
    # ...and the supporting glossary terms.
    glossary_context = state.get("supporting_context", [])

    # Information for debugging
    # logging.info(f"DEBUG: Explanation agent received {len(evidence_context)} snippets to process.")

    # Log the exact evidence the agent is starting with.
    evidence_sources = [Path(s['source_document']).name for s in evidence_context]
    glossary_sources = [s['source_document'] for s in glossary_context]
    logging.info(f"Agent received {len(evidence_sources)} evidence snippets: {evidence_sources}")
    logging.info(f"Agent received {len(glossary_sources)} support snippets: {glossary_sources}")

    # Filter out any glossary items from the main evidence list.
    # This acts as a guard-rail to enforce the prompt's instructions.
    # This guard-rail ensures glossary items are never used as citable evidence.
    usable_evidence = [
        s for s in evidence_context
        if not s.get("support_only") and s.get("source_type") != "bpm-kb"
    ]

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
    
    # Format both context lists for use in the prompts.
    formatted_evidence = format_context_for_prompt(usable_evidence)
    formatted_glossary = format_context_for_prompt(glossary_context)

    # Dynamically select the appropriate prompt based on the drift type.
    drift_type = drift_info.get('drift_type', '').lower()
    if 'sudden' in drift_type:
        draft_prompt_template = SUDDEN_DRIFT_PROMPT
    elif 'gradual' in drift_type:
        draft_prompt_template = GRADUAL_DRIFT_PROMPT
    elif 'recurring' in drift_type:
        draft_prompt_template = RECURRING_DRIFT_PROMPT
    else:
        draft_prompt_template = INCREMENTAL_DRIFT_PROMPT
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(ExplanationOutput)
    llm_cache = load_cache()
    cache_updated = False
    
    try:
        # --- STEP 1: Generate the Draft ---
        draft_prompt = draft_prompt_template.format(
            drift_type=drift_info['drift_type'],
            start_timestamp=drift_info['start_timestamp'],
            end_timestamp=drift_info['end_timestamp'],
            formatted_glossary=formatted_glossary,
            formatted_evidence=formatted_evidence
        )
        draft_cache_key = get_cache_key(draft_prompt, MODEL_NAME)
        
        if draft_cache_key in llm_cache:
            logging.info("Step 1: CACHE HIT for draft explanation.")
            draft_explanation_dict = llm_cache[draft_cache_key]
        else:
            logging.info(f"Step 1: CACHE MISS. Generating draft with {drift_type.upper()} prompt...")
            draft_explanation_obj = structured_llm.invoke(draft_prompt)
            draft_explanation_dict = draft_explanation_obj.dict()
            llm_cache[draft_cache_key] = draft_explanation_dict
            cache_updated = True
            logging.info("Draft generated and cached successfully.")

        # --- STEP 2: Critique and Refine the Draft ---
        refine_prompt = REFINE_PROMPT_TEMPLATE.format(
            formatted_glossary=formatted_glossary,
            formatted_evidence=formatted_evidence,
            draft_explanation=json.dumps(draft_explanation_dict, indent=2)
        )
        refine_cache_key = get_cache_key(refine_prompt, MODEL_NAME)
        
        if refine_cache_key in llm_cache:
            logging.info("Step 2: CACHE HIT for refined explanation.")
            final_explanation_dict = llm_cache[refine_cache_key]
        else:
            logging.info("Step 2: CACHE MISS. Refining draft...")
            final_explanation_obj = structured_llm.invoke(refine_prompt)
            final_explanation_dict = final_explanation_obj.dict()
            llm_cache[refine_cache_key] = final_explanation_dict
            cache_updated = True
            logging.info("Successfully synthesized and cached final explanation.")
        
        # Log the summary from the LLM before calibration
        logging.info(f"  > Generated Summary: {final_explanation_dict.get('summary')}")

        # --- STEP 3: Calibrate Confidence Scores ---
        # Calculate data-driven confidence score before calibration
        ranked_causes = final_explanation_dict.get("ranked_causes", [])
        # Use the index `i` to reliably match the cause to its original evidence snippet,
        # as we instruct the LLM to process them in order.
        for i, cause in enumerate(ranked_causes):
            if i < len(usable_evidence):
                original_snippet = usable_evidence[i]
                cause['confidence_score'] = calculate_confidence_score(original_snippet, drift_info, rank=i)
                logging.info(f"Confidence score for '{original_snippet['source_document']}: {cause['confidence_score']:.2f}")
            else:
                # Fallback if the LLM hallucinates an extra cause.
                cause['confidence_score'] = 0.0
  

        final_explanation: Explanation = {
            "summary": final_explanation_dict.get("summary"),
            "ranked_causes": ranked_causes
        }
        
        if cache_updated:
            save_to_cache(llm_cache)
            logging.info("LLM cache file updated.")
            
        return {"explanation": final_explanation}

    except Exception as e:
        logging.error(f"Failed to generate final explanation: {e}", exc_info=True)
        return {"error": f"Failed to generate final explanation: {e}"}