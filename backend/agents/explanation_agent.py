# backend/agents/explanation_agent.py

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


# --- NEW: Specialized Prompt Templates for 4 Drift Types ---

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


def run_explanation_agent(state: GraphState) -> dict:
    """
    Generates a final explanation by synthesizing all gathered information.
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
        return {"explanation": no_context_explanation}

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        return {"error": "OPENAI_API_KEY not found."}
    
    # --- UPDATED: Dynamic Prompt Selection for 4 Drift Types ---
    drift_type = drift_info.get('drift_type', '').lower()
    
    if 'sudden' in drift_type:
        logging.info("Using SUDDEN drift prompt template.")
        prompt_template = SUDDEN_DRIFT_PROMPT
    elif 'gradual' in drift_type:
        logging.info("Using GRADUAL drift prompt template.")
        prompt_template = GRADUAL_DRIFT_PROMPT
    elif 'recurring' in drift_type:
        logging.info("Using RECURRING drift prompt template.")
        prompt_template = RECURRING_DRIFT_PROMPT
    else: # Default to incremental for 'incremental' or any other type
        logging.info("Using INCREMENTAL drift prompt template.")
        prompt_template = INCREMENTAL_DRIFT_PROMPT
    
    formatted_context = format_context_for_prompt(classified_context)
    prompt = prompt_template.format(
        drift_type=drift_info['drift_type'],
        start_timestamp=drift_info['start_timestamp'],
        end_timestamp=drift_info['end_timestamp'],
        formatted_context=formatted_context
    )
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(ExplanationOutput)

    try:
        logging.info("Synthesizing final explanation...")
        explanation_obj = structured_llm.invoke(prompt)

        final_explanation: Explanation = {
            "summary": explanation_obj.summary,
            "ranked_causes": [cause.dict() for cause in explanation_obj.ranked_causes]
        }
        logging.info("Successfully synthesized explanation.")
        return {"explanation": final_explanation}

    except Exception as e:
        logging.error(f"Failed to generate final explanation: {e}")
        return {"error": f"Failed to generate final explanation: {e}"}