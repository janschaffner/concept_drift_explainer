import os
import sys
import logging
from pathlib import Path
from typing import List, Dict
from enum import Enum

# --- Path Correction ---
# Ensures that the script can correctly import modules from the 'backend' directory.
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
# -----------------------

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic.v1 import BaseModel, Field
from backend.utils.cache import load_cache, save_to_cache, get_cache_key

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
MODEL_NAME = "gpt-4o-mini"

class ConnectionType(str, Enum):
    """Enum for the different types of connections between drifts."""
    STRONG_CAUSAL = "Strong Causal Link"
    SHARED_EVIDENCE = "Shared Evidence / Root Cause"
    THEMATIC_OVERLAP = "Thematic Overlap"
    NONE = "No Significant Connection"

# --- Pydantic Model for Structured Output ---
# This model forces the LLM to make a boolean decision before generating text,
# which prevents it from hallucinating connections where none exist.
class DriftLinkAnalysis(BaseModel):
    """Data model for the drift relationship analysis."""
    connection_type: ConnectionType = Field(description="The classification of the relationship between the drifts.")
    summary: str = Field(description="Your detailed analysis and reasoning for the chosen connection type. If no connection is found, briefly state that.")

def format_states_for_prompt(full_states: List[Dict]) -> str:
    """
    Formats a list of full state objects into a structured string for the LLM prompt.
    
    This function creates a detailed profile of each drift, including its timeframe,
    summary, all evidence sources, and specific entities to be used for meta-analysis.

    Args:
        full_states: A list of full state dictionaries from each pipeline run.

    Returns:
        A formatted string detailing the structured data for each drift.
    """
    formatted_str = ""
    for i, state in enumerate(full_states):
        if not state: continue
        
        drift_info = state.get('drift_info', {})
        explanation = state.get('explanation', {})
        specific_entities = state.get('specific_entities', [])
        ranked_causes = explanation.get('ranked_causes', [])
        evidence_sources = sorted(list(set(c.get('source_document', 'N/A') for c in ranked_causes)))
        
        formatted_str += f"### Drift #{i+1} Profile\n"
        formatted_str += f"- **Type:** {drift_info.get('drift_type', 'N/A')}\n"
        formatted_str += f"- **Timeframe:** {drift_info.get('start_timestamp')} to {drift_info.get('end_timestamp')}\n"
        formatted_str += f"- **Summary:** {explanation.get('summary', 'N/A')}\n"
        formatted_str += f"- **Evidence Sources:** {evidence_sources}\n"
        formatted_str += f"- **Specific Entities:** {specific_entities[:5]}\n" # Show top 5 entities
        formatted_str += "\n"

    return formatted_str

def run_drift_linker_agent(full_states: List[Dict]) -> dict:
    """
    Analyzes a list of drift explanations to find potential relationships between them.

    This agent performs a meta-analysis on the outputs of the main pipeline. It is
    called once at the end of a batch run to provide a high-level summary of how
    different drifts might be interconnected.

    Note: This agent is called directly by the UI and does not use the GraphState.

    Args:
        full_states: A list of all final state objects from each pipeline run.

    Returns:
        A dictionary containing the `linked_drift_summary`.
    """
    logging.info("--- Running Drift Linker Agent ---")
    
    if not full_states or len(full_states) < 2:
        logging.warning("Not enough explanations to run relationship analysis.")
        return {"linked_drift_summary": None}

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        return {"error": "OPENAI_API_KEY not found."}
        
    # Format the inputs for the prompt.
    formatted_context = format_states_for_prompt(full_states)
    
    # --- Prompt Template ---
    # This prompt asks the LLM to act as a senior analyst and find connections.
    prompt_template = """You are a senior business process analyst conducting a meta-analysis. Your goal is to find high-level insights by identifying potential relationships between several independently explained concept drifts.

    You will be provided with structured data profiles for multiple concept drifts that occurred in the same process log.

    **## Drift Profiles**
    {formatted_context}

    **## Your Task**
    Carefully review the structured data for all drift profiles. Analyze their timeframes, summaries, evidence sources, and specific entities to identify potential connections.
    First, classify the connection type by choosing one of the following options: 'Strong Causal Link', 'Shared Evidence / Root Cause', 'Thematic Overlap', or 'No Significant Connection'.    
    Then, provide a detailed summary explaining your reasoning for the chosen connection type.
    """
    
    prompt = prompt_template.format(formatted_context=formatted_context)
    
    # Use the structured output method to force a yes/no decision.
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
    structured_llm = llm.with_structured_output(DriftLinkAnalysis)
    
    # Caching Logic
    llm_cache = load_cache()
    cache_key = get_cache_key(prompt, MODEL_NAME)

    if cache_key in llm_cache:
        logging.info("CACHE HIT for drift relationship analysis.")
        response_data = llm_cache[cache_key]
    else:
        logging.info("CACHE MISS. Calling API for drift relationship analysis...")
        try:
            response_object = structured_llm.invoke(prompt)
            response_data = response_object.dict()
            llm_cache[cache_key] = response_data
            save_to_cache(llm_cache)
            logging.info("Drift link analysis cached successfully.")
        except Exception as e:
            logging.error(f"Failed to generate linked drift summary: {e}")
            return {"error": str(e)}

    # Only return the summary if the LLM explicitly found a connection.
    connection_type = response_data.get("connection_type")
    if connection_type and connection_type != ConnectionType.NONE.value:
        summary = response_data.get("summary")
        # This log message shows the exact summary that will be displayed to the user.
        logging.info(f"Connection between drifts found. Type: '{connection_type}'.")
        return {"linked_drift_summary": summary, "connection_type": connection_type}
    else:
        logging.info("No significant connection found between drifts.")
        return {"linked_drift_summary": None}