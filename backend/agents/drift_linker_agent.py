import os
import sys
import logging
from pathlib import Path
from typing import List, Dict

# --- Path Correction ---
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

# --- Pydantic Model for Structured Output ---
class DriftLinkAnalysis(BaseModel):
    """Data model for the drift relationship analysis."""
    connection_found: bool = Field(description="Set to true ONLY if there is a clear, evidence-based connection between the drifts. Otherwise, set to false.")
    summary: str = Field(description="If connection_found is true, describe the connection here. Otherwise, state that no significant connection was identified.")

def format_explanations_for_prompt(all_explanations: List[Dict]) -> str:
    """Formats the list of full explanation objects into a string for the LLM prompt."""
    formatted_str = ""
    for i, explanation in enumerate(all_explanations):
        if not explanation: continue
        
        summary = explanation.get('summary', 'N/A')
        ranked_causes = explanation.get('ranked_causes', [])
        
        formatted_str += f"### Explanation for Drift #{i+1}\n"
        formatted_str += f"- **Summary:** {summary}\n"
        
        if ranked_causes:
            top_cause = ranked_causes[0]
            formatted_str += f"- **Top Cause:** {top_cause.get('cause_description', 'N/A')}\n"
            formatted_str += f"- **Top Evidence Source:** {top_cause.get('source_document', 'N/A')}\n"
        formatted_str += "\n"
        
    return formatted_str

def run_drift_linker_agent(all_explanations: List[Dict]) -> dict:
    """
    Analyzes a list of explanations to find relationships between them.
    Note: This agent is called directly and does not use the GraphState.
    """
    logging.info("--- Running Drift Linker Agent ---")
    
    if not all_explanations or len(all_explanations) < 2:
        logging.warning("Not enough explanations to run relationship analysis.")
        return {"linked_drift_summary": ""}

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        return {"error": "OPENAI_API_KEY not found."}
        
    # Format the inputs for the prompt
    formatted_explanations = format_explanations_for_prompt(all_explanations)
    
    # --- Prompt Template ---
    prompt_template = """You are a senior business process analyst conducting a meta-analysis. Your goal is to find high-level insights by identifying potential relationships between several independently explained concept drifts.

    You will be provided with a list of explanations for multiple concept drifts that occurred in the same process log.

    **## Explained Drifts**
    {formatted_explanations}

    **## Your Task**
    Carefully review all the provided drift explanations. First, decide if there is a clear, evidence-based connection between any of the drifts. A connection exists if they share a common root cause, cite the same source document, or one directly causes another.
    
    Then, provide your analysis using the following structured format.
    """
    
    prompt = prompt_template.format(formatted_explanations=formatted_explanations)
    
    # Use the structured output method to force a yes/no decision
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

    # Only return the summary if a connection was explicitly found
    if response_data.get("connection_found"):
        summary = response_data.get("summary")
        # This log message shows the exact summary that will be displayed to the user.
        logging.info(f"Connection between drifts found. Summary: '{summary}'")
        return {"linked_drift_summary": summary}
    else:
        logging.info("No significant connection found between drifts.")
        return {"linked_drift_summary": None}