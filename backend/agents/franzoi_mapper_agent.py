# context_drift_explainer/backend/agents/franzoi_mapper_agent.py

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
from backend.state.schema import GraphState, ContextSnippet, FranzoiClassification
# Import the caching utility
from backend.utils.cache import load_cache, save_to_cache, get_cache_key

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
MODEL_NAME = "gpt-4o-mini"

# --- Pydantic Models for Structured Output ---
# This defines the structure of a single classification.
# It's like a schema for the LLM's "tool".
class Classification(BaseModel):
    full_path: str = Field(description="The full hierarchical path of the category, e.g., ORGANIZATION_INTERNAL::Process_Management")
    reasoning: str = Field(description="A brief reasoning for this classification.")

# This defines the top-level object the LLM should produce.
class ClassificationList(BaseModel):
    """A list of all relevant classifications for the given text snippet."""
    classifications: List[Classification]

# The prompt is now simpler, as the complex instructions are in the tool definition.
PROMPT_TEMPLATE = """You are an expert business process analyst specializing in process mining. Your task is to classify a given text snippet against the full three-level Franzoi et al. context taxonomy.

The snippet may fit into one or more categories. Identify all relevant categories from the taxonomy provided below.

TAXONOMY:
- LEVEL 1: PROCESS_IMMEDIATE (data directly related to the process execution)
  - Case: Properties of a single process instance (e.g., case ID, type of product).
  - Resource: Information about the actor performing the work (e.g., skill level, workload).
  - System_Interaction: Technical system events (e.g., button clicks, API calls).
- LEVEL 2: ORGANIZATION_INTERNAL (context from within the organization)
  - Organizational: Restructuring, new roles, departmental changes.
  - Process_Management: New KPIs, process redesign, new Standard Operating Procedures (SOPs).
  - IT_Management: New software rollout, system updates, server maintenance.
- LEVEL 3: ORGANIZATION_EXTERNAL (context from outside the organization)
  - Economic: Market shifts, competitor actions, price changes.
  - Social: Public holidays, seasonality, social trends, pandemics.
  - Legal: New laws, regulations, compliance requirements.
  - Technical: New external standards, infrastructure changes (e.g., cloud provider outage).

TEXT SNIPPET:
"{snippet_text}"
"""

def run_franzoi_mapper_agent(state: GraphState) -> dict:
    """
    Classifies retrieved context snippets using the Franzoi taxonomy and
    updates them in-place within the state.
    """
    logging.info("--- Running Franzoi Mapper Agent (Tool-Calling Version) ---")

    # NOTE: We now get the list and modify it directly.
    context_snippets: List[ContextSnippet] = state.get("raw_context_snippets", [])
    if not context_snippets:
        logging.warning("No context snippets found to classify.")
        return {} # Return empty dict as we are done

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        error_msg = "OPENAI_API_KEY not found in .env file."
        logging.error(error_msg)
        return {"error": error_msg}

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(ClassificationList)

    # --- NEW: Load the cache at the start of the process ---
    llm_cache = load_cache()
    cache_updated = False

    for snippet in context_snippets: # Loop through and modify each snippet
        logging.info(f"Classifying snippet from: {snippet['source_document']}")

        prompt = PROMPT_TEMPLATE.format(snippet_text=snippet["snippet_text"])

        # --- NEW: Caching Logic ---
        cache_key = get_cache_key(prompt, MODEL_NAME)
        
        if cache_key in llm_cache:
            logging.info(f"CACHE HIT for snippet from: {snippet['source_document']}")
            response_data = llm_cache[cache_key]
        else:
            logging.info(f"CACHE MISS. Calling API for snippet from: {snippet['source_document']}")
            try:
                response_object = structured_llm.invoke(prompt)
                # Convert Pydantic object to a standard dictionary to store in JSON cache
                response_data = response_object.dict()
                
                # Save the new response to the cache
                llm_cache[cache_key] = response_data
                cache_updated = True
            except Exception as e:
                logging.error(f"Error classifying snippet: {e}")
                snippet['classifications'] = [{"full_path": "CLASSIFICATION_FAILED", "reasoning": str(e)}]
                continue

        # Process the response data (either from cache or new API call)
        typed_classifications: List[FranzoiClassification] = [
            {"full_path": item.get("full_path", "UNKNOWN"), "reasoning": item.get("reasoning", "")}
            for item in response_data.get("classifications", [])
        ]
        snippet['classifications'] = typed_classifications
        logging.info(f"  > Classified as: {[c['full_path'] for c in typed_classifications]}")
        
    # --- NEW: Save the cache if it has been updated ---
    if cache_updated:
        save_to_cache(llm_cache)
        logging.info("LLM cache file updated.")


    logging.info("Franzoi Mapper Agent execution successful.")

    # We modified the state in-place, so we return an empty dictionary.
    return {}