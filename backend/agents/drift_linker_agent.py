"""
This module contains the implementation of the Drift Linker Agent.

This agent performs a meta-analysis on the results of multiple, individual drift
explanations. It is invoked by the frontend after the main pipeline has been run
for two or more drifts. Its purpose is to identify and summarize high-level
relationships between these drifts, such as a shared root cause or a causal link,
providing an additional layer of analytical insight.
"""

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
MAX_DRIFTS_TO_LINK = 10 # Hard limit for the number of drifts to analyze to prevent overly long prompts

# --- Pydantic Models for Structured Output ---
class ConnectionType(str, Enum):
    """Enum for the different types of connections between drifts."""
    STRONG_CAUSAL = "Strong Causal Link"
    SHARED_EVIDENCE = "Shared Evidence / Root Cause"
    THEMATIC_OVERLAP = "Thematic Overlap"
    NONE = "No Significant Connection"

class DriftLinkAnalysis(BaseModel):
    """Data model for the drift relationship analysis."""
    connection_type: ConnectionType = Field(description="The classification of the relationship between the drifts.")
    summary: str = Field(description="Your detailed analysis and reasoning for the chosen connection type. If no connection is found, briefly state that.")

# --- Helper Functions ---

def format_states_for_prompt(full_states: List[Dict]) -> str:
    """Formats a list of state objects into a structured string for the LLM.

    This function creates a detailed, human-readable profile for each drift,
    extracting its key characteristics (type, timeframe, summary, evidence) to
    make it easy for the LLM to compare and analyze them.

    Args:
        full_states: A list of final GraphState dictionaries for each drift.

    Returns:
        A formatted string containing the profiles of all drifts.
    """
    formatted = ""
    for i, st in enumerate(full_states, 1):
        di = st["drift_info"]
        ex = st["explanation"]
        # Pull in the drift_phrase, which is the key semantic signal
        drift_phrase = st.get("drift_phrase", "N/A")

        # Use only the top-3 causes for brevity in the prompt.
        top_causes = ex.get("ranked_causes", [])[:3]
        evidence_docs = sorted({c["source_document"] for c in top_causes})
        
        formatted += f"### Drift #{i}\n"
        formatted += f"- Type: {di.get('drift_type', 'N/A')}\n"
        formatted += f"- Timeframe: {di.get('start_timestamp')} to {di.get('end_timestamp')}\n"
        formatted += f"- Summary: {ex.get('summary', 'N/A')}\n"
        formatted += f"- Evidence Docs: {evidence_docs}\n"
        formatted += f"- Key Semantic Phrase: \"{drift_phrase}\"\n"
        formatted += f"- Top Causes:\n"
        for c in top_causes:
            formatted += (
                f"    • {c.get('cause_description', 'N/A')} "
                f"(Category={c.get('context_category', 'N/A')}, "
                f"Conf={c.get('confidence_score', 0.0):.2f})\n"
            )
        formatted += "\n"
    return formatted

# --- Main Agent Logic ---

def run_drift_linker_agent(full_states: List[Dict]) -> dict:
    """Analyzes a list of drift explanations to find potential relationships.

    This agent is run as a post-processing step by the frontend. It takes the
    final states of all analyzed drifts, formats them into a comprehensive
    prompt, and uses an LLM to perform a meta-analysis to classify the
    relationship between them.

    Args:
        full_states: A list of final GraphState objects from the main pipeline.

    Returns:
        A dictionary containing the `connection_type` and `linked_drift_summary`.
    """
    logging.info("--- Running Drift Linker Agent ---")
    
    # Step 1: Input validation. The agent requires at least two drifts to perform
    # a meaningful comparison.
    if not full_states or len(full_states) < 2:
        logging.warning("Need at least two drifts to run relationship analysis.")
        # Always return a consistent schema
        return {"linked_drift_summary": None, "connection_type": ConnectionType.NONE.value}

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        return {"error": "OPENAI_API_KEY not found."}
    
    # Step 2: Prepare the input for the LLM. Truncate the list of drifts to
    # the most recent N to manage prompt length and API costs.
    states_to_process = sorted(full_states, key=lambda s: s.get("drift_info", {}).get("start_timestamp", ""), reverse=True)
    states_to_process = states_to_process[:MAX_DRIFTS_TO_LINK]
    
    # Add a log message to show what the agent is doing
    drift_types_to_link = [s.get("drift_info", {}).get("drift_type", "Unknown") for s in states_to_process]
    logging.info(f"Analyzing relationships between {len(states_to_process)} drifts: {drift_types_to_link}")

    # Format the inputs for the prompt.
    context_block = format_states_for_prompt(states_to_process)
    
    # Step 3: Build the prompt dynamically and make the cached LLM call.
    # The prompt instructs the LLM to act as a senior analyst performing a meta-analysis.
    connection_options = "\n".join([f"- `{e.value}`" for e in ConnectionType])
    prompt = f"""You are a senior business process analyst conducting a meta-analysis. 
    Your goal is to find high-level insights by identifying potential relationships between several independently explained concept drifts.
    You will be provided with structured data profiles for multiple concept drifts that occurred in the same process log.

    **## Drift Profiles**
    {context_block}

    **## Your Task**
    1. Carefully review the structured data for all drift profiles. Analyze their timeframes, summaries, evidence sources, and Key Semantic Phrase to identify potential connections.
    2. First, classify the connection type by choosing **only one** of the following options:
    {connection_options}
    3. Then, provide a detailed, 1-2 sentence summary explaining your reasoning for the chosen connection type.
    """
    
    # Use the structured output method to force a yes/no decision.
    logging.info("Building LLM prompt for drift-link analysis")
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
        
    # Step 4: Process the response and return it in a consistent format. If no
    # significant connection is found, the summary is set to None.
    connection_type = response_data.get("connection_type", ConnectionType.NONE.value)
    summary = response_data.get("summary") if connection_type != ConnectionType.NONE.value else None

    # Final log message to clearly state the result
    if summary:
        logging.info(f"✅ Relationship Found: {connection_type}")
    else:
        logging.info("❌ No significant relationship found between drifts.")
        
    return {
        "connection_type": connection_type,
        "linked_drift_summary": summary
    }