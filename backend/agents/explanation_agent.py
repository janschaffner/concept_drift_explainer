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
from backend.state.schema import GraphState, Explanation, RankedCause

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Pydantic Models for Structured Output ---
class Cause(BaseModel):
    """Represents a single, ranked potential cause for the drift."""
    cause_description: str = Field(description="The detailed analysis of what caused the drift, citing the evidence.")
    evidence_snippet: str = Field(description="The specific text snippet that supports the analysis.")
    source_document: str = Field(description="The name of the source document for the evidence.")
    context_category: str = Field(description="The most relevant Franzoi context category path.")
    confidence_score: float = Field(description="Confidence in this cause, from 0.0 to 1.0.")

class ExplanationOutput(BaseModel):
    """The final, structured explanation for the concept drift."""
    summary: str = Field(description="A 1-3 sentence executive summary of the most likely cause.")
    ranked_causes: List[Cause] = Field(description="A list of potential causes, ordered from most to least likely.")


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
        # Create a default explanation indicating no context was found
        no_context_explanation: Explanation = {
            "summary": "No explanation could be generated as no relevant contextual documents were found for the detected drift period.",
            "ranked_causes": []
        }
        return {"explanation": no_context_explanation}

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        error_msg = "OPENAI_API_KEY not found in .env file."
        logging.error(error_msg)
        return {"error": error_msg}
    
    # Format the inputs for the prompt
    formatted_context = format_context_for_prompt(classified_context)
    
    prompt_template = """You are an expert business process analyst and a skilled technical writer. Your goal is to explain a detected concept drift to a manager in a clear, concise, and actionable way.

You will be given information about the detected drift and a list of potentially relevant text snippets that have been automatically classified according to the Franzoi et al. context taxonomy.

**## 1. Detected Concept Drift**
- **Drift Type:** {drift_type}
- **Drift Period:** {start_timestamp} to {end_timestamp}
- **Confidence:** {confidence:.2f}

**## 2. Evidence from Context Documents**
{formatted_context}

**## 3. Your Task**
Based on all the provided information, generate a final explanation. Structure your response as a valid JSON object with two keys: "summary" and "ranked_causes".

- **"summary"**: A 1-3 sentence executive summary of the most likely cause of the drift.
- **"ranked_causes"**: A list of potential causes, ordered from most to least likely. Each cause in the list should be a JSON object with the following keys:
  - **"cause_description"**: Your analysis of what caused the drift.
  - **"evidence_snippet"**: The specific text snippet that supports your analysis.
  - **"source_document"**: The name of the source document for the evidence.
  - **"context_category"**: The most relevant classification path for this cause (e.g., "ORGANIZATION_INTERNAL::Process_Management").
  - **"confidence_score"**: Your confidence in this cause, from 0.0 to 1.0.

Your tone should be objective and professional. Use phrases like "The evidence suggests..." or "A potential cause is...".
"""
    
    prompt = prompt_template.format(
        drift_type=drift_info['drift_type'],
        start_timestamp=drift_info['start_timestamp'],
        end_timestamp=drift_info['end_timestamp'],
        confidence=drift_info['confidence'],
        formatted_context=formatted_context
    )
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    structured_llm = llm.with_structured_output(ExplanationOutput)

    try:
        logging.info("Synthesizing final explanation...")
        explanation_obj = structured_llm.invoke(prompt)

        # Convert from Pydantic models to TypedDicts for the state
        final_explanation: Explanation = {
            "summary": explanation_obj.summary,
            "ranked_causes": [cause.dict() for cause in explanation_obj.ranked_causes]
        }
        logging.info("Successfully synthesized explanation.")
        return {"explanation": final_explanation}

    except Exception as e:
        logging.error(f"Failed to generate final explanation: {e}")
        return {"error": f"Failed to generate final explanation: {e}"}