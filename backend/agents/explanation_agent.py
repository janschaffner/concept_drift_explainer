# backend/agents/explanation_agent.py

import os
import sys
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

# Import our graph state schema
from backend.state.schema import GraphState, Explanation

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

**## 2. Evidence from Context Documents**
{formatted_context}

**## 3. Your Task**
Based on the provided information, generate an explanation for the **SUDDEN** drift. Structure your response as a valid JSON object with "summary" and "ranked_causes" keys.
- **"summary"**: A 1-3 sentence executive summary of the most likely cause.
- **"ranked_causes"**: A list of potential causes, ordered from most likely to least likely. Focus on singular events.
"""

GRADUAL_DRIFT_PROMPT = """You are an expert business process analyst. Your goal is to explain a **Gradual Drift**.
A Gradual Drift describes a transition phase where both the old and new process variants coexist. This is common in rollout scenarios, where a new process is adopted for new cases, while ongoing cases continue under the old variant. Over time, the older version is phased out entirely (Bose et al., 2011).
Prioritize evidence suggesting a transition, coexistence of old/new processes, or phased rollouts.

**## 1. Detected Concept Drift**
- **Drift Type:** {drift_type}
- **Drift Period:** {start_timestamp} to {end_timestamp}

**## 2. Evidence from Context Documents**
{formatted_context}

**## 3. Your Task**
Based on the provided information, generate an explanation for the **GRADUAL** drift. Structure your response as a valid JSON object with "summary" and "ranked_causes" keys.
- **"summary"**: A 1-3 sentence executive summary of the most likely cause.
- **"ranked_causes"**: A list of potential causes, ordered from most likely to least likely. Focus on transition periods.
"""

INCREMENTAL_DRIFT_PROMPT = """You are an expert business process analyst. Your goal is to explain an **Incremental Drift**.
An Incremental Drift consists of a sequence of small, continuous changes that cumulatively result in significant process transformation. It is often associated with agile BPM practices, where iterative adjustments are made without a single, identifiable change point (Bose et al., 2011; Kraus und van der Aa, 2025).
Prioritize evidence of multiple small adjustments, iterative improvements, or agile practices over time.

**## 1. Detected Concept Drift**
- **Drift Type:** {drift_type}
- **Drift Period:** {start_timestamp} to {end_timestamp}

**## 2. Evidence from Context Documents**
{formatted_context}

**## 3. Your Task**
Based on the provided information, generate an explanation for the **INCREMENTAL** drift. Structure your response as a valid JSON object with "summary" and "ranked_causes" keys.
- **"summary"**: A 1-3 sentence executive summary of the most likely cause.
- **"ranked_causes"**: A list of potential causes, ordered from most likely to least likely. Focus on a series of small changes.
"""

RECURRING_DRIFT_PROMPT = """You are an expert business process analyst. Your goal is to explain a **Recurring Drift**.
A Recurring Drift occurs when previously observed process versions reappear over time, often in a cyclical pattern. These drifts may follow seasonal cycles or non-periodic triggers (e.g., market-specific promotional workflows) (Bose et al., 2011; Kraus und van der Aa, 2025).
Prioritize evidence of seasonal activities, cyclical patterns, or temporary process changes that are designed to reappear.

**## 1. Detected Concept Drift**
- **Drift Type:** {drift_type}
- **Drift Period:** {start_timestamp} to {end_timestamp}

**## 2. Evidence from Context Documents**
{formatted_context}

**## 3. Your Task**
Based on the provided information, generate an explanation for the **RECURRING** drift. Structure your response as a valid JSON object with "summary" and "ranked_causes" keys.
- **"summary"**: A 1-3 sentence executive summary of the most likely cause.
- **"ranked_causes"**: A list of potential causes, ordered from most likely to least likely. Focus on cyclical or seasonal evidence.
"""

# --- REFINE PROMPT: A single, generic prompt for refinement ---

REFINE_PROMPT_TEMPLATE = """You are a senior editor reviewing an analysis from a junior analyst.
Your task is to critique and refine the provided "Draft Explanation" based on the original "Evidence".
Ensure the final summary is concise, the cause descriptions are logical, and that every claim is strongly supported by the cited evidence.
Produce a final, improved version of the explanation.

**## Original Evidence**
{formatted_context}

**## Draft Explanation to Review**
{draft_explanation}

**## 3. Your Task**
Generate the final, high-quality version of the explanation. Your output MUST be a valid JSON object in the same format as the draft, with "summary" and "ranked_causes" keys.
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
    Generates a final explanation using a two-step "draft and refine" chain,
    with drift-type-specific logic for the draft stage.
    """
    logging.info("--- Running Explanation Agent ---")
    
    drift_info = state.get("drift_info")
    classified_context = state.get("raw_context_snippets", [])

    if not classified_context:
        logging.warning("No classified context found. Cannot generate an explanation.")
        no_context_explanation: Explanation = {
            "summary": "No explanation could be generated as no relevant contextual documents were found for the detected drift period.",
            "ranked_causes": []
        }
        return {"explanation": {"summary": "No explanation could be generated...", "ranked_causes": []}}

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        return {"error": "OPENAI_API_KEY not found."}
    
    # --- Dynamic Prompt Selection for 4 Drift Types ---
    drift_type = drift_info.get('drift_type', '').lower()
    
    if 'sudden' in drift_type:
        logging.info("Using SUDDEN drift prompt template.")
        draft_prompt_template = SUDDEN_DRIFT_PROMPT
    elif 'gradual' in drift_type:
        logging.info("Using GRADUAL drift prompt template.")
        draft_prompt_template = GRADUAL_DRIFT_PROMPT
    elif 'recurring' in drift_type:
        logging.info("Using RECURRING drift prompt template.")
        draft_prompt_template = RECURRING_DRIFT_PROMPT
    else: # Default to incremental for 'incremental' or any other type
        logging.info("Using INCREMENTAL drift prompt template.")
        draft_prompt_template = INCREMENTAL_DRIFT_PROMPT
    
    formatted_context = format_context_for_prompt(classified_context)
    draft_prompt = draft_prompt_template.format(
        drift_type=drift_info['drift_type'],
        start_timestamp=drift_info['start_timestamp'],
        end_timestamp=drift_info['end_timestamp'],
        formatted_context=formatted_context
    )
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(ExplanationOutput)

    try:
        # === STEP 1: Generate the Draft ===
        logging.info(f"Step 1: Generating draft explanation with {drift_type.upper()} prompt...")
        draft_explanation_obj = structured_llm.invoke(draft_prompt)
        logging.info("Draft generated successfully.")

        # === STEP 2: Critique and Refine the Draft ===
        logging.info("Step 2: Refining draft with self-correction prompt...")
        refine_prompt = REFINE_PROMPT_TEMPLATE.format(
            formatted_context=formatted_context,
            draft_explanation=draft_explanation_obj.json()
        )
        final_explanation_obj = structured_llm.invoke(refine_prompt)
        logging.info("Successfully synthesized final explanation.")

        # Convert Pydantic object to a list of dictionaries
        ranked_causes_dicts = [cause.dict() for cause in final_explanation_obj.ranked_causes]

        # === STEP 3: Calibrate Confidence Scores ===
        calibrated_causes = calibrate_scores(ranked_causes_dicts, drift_info)

        final_explanation: Explanation = {
            "summary": final_explanation_obj.summary,
            "ranked_causes": [cause.dict() for cause in final_explanation_obj.ranked_causes]
        }
        return {"explanation": final_explanation}

    except Exception as e:
        logging.error(f"Failed to generate final explanation: {e}")
        return {"error": f"Failed to generate final explanation: {e}"}