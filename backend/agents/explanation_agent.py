import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict
from datetime import datetime

# --- Path Correction ---
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
class Cause(BaseModel):
    cause_description: str = Field(description="The detailed analysis of what caused the drift, citing the evidence.")
    evidence_snippet: str = Field(description="The specific text snippet that supports the analysis.")
    source_document: str = Field(description="The name of the source document for the evidence.")
    context_category: str = Field(description="The most relevant Franzoi context category path.")
    confidence_score: float = Field(description="Confidence in this cause, from 0.0 to 1.0.")

class ExplanationOutput(BaseModel):
    summary: str = Field(description="A 1-3 sentence executive summary of the most likely cause.")
    ranked_causes: List[Cause] = Field(description="A list of potential causes, ordered from most to least likely.")


# --- DRIFT PROMPTS: Specialized Prompt Templates for 4 Drift Types ---

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
Based on the provided information, generate an explanation for the **SUDDEN** drift. Cite only documents listed in the Evidence section; do NOT cite glossary items as formal evidence. Structure your response as a valid JSON object with "summary" and "ranked_causes" keys.
- **"summary"**: A 1-3 sentence executive summary of the most likely cause.
- **"ranked_causes"**: A list of potential causes, ordered from most likely to least likely. Focus on singular events.
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
Based on the provided information, generate an explanation for the **GRADUAL** drift. Cite only documents listed in the Evidence section; do NOT cite glossary items as formal evidence. Structure your response as a valid JSON object with "summary" and "ranked_causes" keys.
- **"summary"**: A 1-3 sentence executive summary of the most likely cause.
- **"ranked_causes"**: A list of potential causes, ordered from most likely to least likely. Focus on transition periods.
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
Based on the provided information, generate an explanation for the **INCREMENTAL** drift. Cite only documents listed in the Evidence section; do NOT cite glossary items as formal evidence. Structure your response as a valid JSON object with "summary" and "ranked_causes" keys.
- **"summary"**: A 1-3 sentence executive summary of the most likely cause.
- **"ranked_causes"**: A list of potential causes, ordered from most likely to least likely. Focus on a series of small changes.
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
Based on the provided information, generate an explanation for the **RECURRING** drift. Cite only documents listed in the Evidence section; do NOT cite glossary items as formal evidence. Structure your response as a valid JSON object with "summary" and "ranked_causes" keys.
- **"summary"**: A 1-3 sentence executive summary of the most likely cause.
- **"ranked_causes"**: A list of potential causes, ordered from most likely to least likely. Focus on cyclical or seasonal evidence.
"""

# --- REFINE PROMPT: A single, generic prompt for refinement ---
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
Generate the final, high-quality version of the explanation. Cite only documents listed in the Evidence section; do NOT cite glossary items as formal evidence. Your output MUST be a valid JSON object in the same format as the draft, with "summary" and "ranked_causes" keys.
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


# --- Confidence Score Calibration Logic ---
def get_timestamp_from_filename(filename: str) -> int:
    """Parses YYYY-MM-DD from the start of a filename and returns a Unix timestamp."""
    try:
        date_str = filename.split('_')[0]
        dt_obj = datetime.strptime(date_str, '%Y-%m-%d')
        return int(dt_obj.timestamp())
    except (ValueError, IndexError):
        return 0

def calibrate_scores(ranked_causes: List[Dict], drift_info: Dict) -> List[Dict]:
    """
    Adjusts confidence scores based on custom rules.
    Rule 1: Penalize evidence that is temporally distant from the drift.
    """
    logging.info("Step 3: Calibrating confidence scores...")
    
    TIME_THRESHOLD_DAYS = 60  # Evidence older than this will be penalized
    PENALTY_FACTOR = 0.75     # Score will be multiplied by this factor (25% reduction)

    drift_start_dt = datetime.fromisoformat(drift_info["start_timestamp"])
    
    calibrated_causes = []
    for cause in ranked_causes:
        evidence_ts = get_timestamp_from_filename(cause["source_document"])
        if evidence_ts > 0:
            evidence_dt = datetime.fromtimestamp(evidence_ts)
            delta_days = abs((drift_start_dt - evidence_dt).days)
            
            if delta_days > TIME_THRESHOLD_DAYS:
                original_score = cause["confidence_score"]
                new_score = original_score * PENALTY_FACTOR
                cause["confidence_score"] = round(new_score, 2)
                logging.info(
                    f"  > Penalizing cause from '{cause['source_document']}'. "
                    f"Temporal distance: {delta_days} days. "
                    f"Score reduced from {original_score:.2f} to {new_score:.2f}."
                )
        calibrated_causes.append(cause)
        
    return calibrated_causes


def run_explanation_agent(state: GraphState) -> dict:
    """
    Generates and then calibrates a final explanation using a two-step "draft and refine" chain.
    """
    logging.info("--- Running Explanation Agent (Self-Correction Chain) ---")
    
    drift_info = state.get("drift_info")
    # Use the refined list from the re-ranker
    evidence_context = state.get("reranked_context_snippets", [])
    # Get the supporting glossary terms
    glossary_context = state.get("supporting_context", [])

    # --- Filter out any glossary items from the main evidence list ---
    # This acts as a guard-rail to enforce the prompt's instructions.
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
    
    # --- Format two separate context strings ---
    # UPDATED: Use the filtered 'usable_evidence' list
    formatted_evidence = format_context_for_prompt(usable_evidence)
    formatted_glossary = format_context_for_prompt(glossary_context)

    # Dynamic Prompt Selection logic remains the same...
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
        # === STEP 1: Generate the Draft ===
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

        # === STEP 2: Critique and Refine the Draft ===
        # CORRECTED: The .format() call now uses the correct variable names
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

        # === STEP 3: Calibrate Confidence Scores ===
        calibrated_causes = calibrate_scores(final_explanation_dict.get("ranked_causes", []), drift_info)

        final_explanation: Explanation = {
            "summary": final_explanation_dict.get("summary"),
            "ranked_causes": calibrated_causes
        }
        
        if cache_updated:
            save_to_cache(llm_cache)
            logging.info("LLM cache file updated.")
            
        return {"explanation": final_explanation}

    except Exception as e:
        logging.error(f"Failed to generate final explanation: {e}", exc_info=True)
        return {"error": f"Failed to generate final explanation: {e}"}