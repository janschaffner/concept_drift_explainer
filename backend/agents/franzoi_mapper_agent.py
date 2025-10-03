"""
This module contains the implementation of the Franzoi Mapper Agent.

Its primary responsibility is to enrich the curated list of evidence snippets by
classifying each one against the formal, three-level Process Mining Context
Taxonomy from Franzoi, Hartl et al. (2025). This agent adds a crucial layer of
structured, academic context to the evidence before it is passed to the final
Explanation Agent.
"""

import os
import sys
import logging
from pathlib import Path
from typing import List

# --- Path Correction ---
# Ensures that the script can correctly import modules from the 'backend' directory.
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
# -----------------------

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic.v1 import BaseModel, Field

# Import our graph state schema and caching utility.
from backend.state.schema import GraphState, ContextSnippet, FranzoiClassification
from backend.utils.cache import load_cache, save_to_cache, get_cache_key

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
MODEL_NAME = "gpt-4o-mini"

# --- Pydantic Models for Structured Output ---
# This defines the structure of a single classification output from the LLM.
class Classification(BaseModel):
    full_path: str = Field(description="The full hierarchical path of the category, e.g., ORGANIZATION_INTERNAL::Process_Management")
    reasoning: str = Field(description="A brief reasoning for this classification.")

# This defines the top-level object the LLM should produce.
class ClassificationList(BaseModel):
    """A list of all relevant classifications for the given text snippet."""
    classifications: List[Classification]

# --- Prompt Template ---
# This prompt operationalizes the Process Mining Context Taxonomy (Franzoi et al., 2025), instructing
# the LLM to act as a process analyst and map a given text snippet to the
# formal, three-level classification system.
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

# --- Main Agent Logic ---

def run_franzoi_mapper_agent(state: GraphState) -> dict:
    """
    Classifies a list of context snippets against the Franzoi context taxonomy.

    This agent iterates through each snippet provided by the Re-Ranker Agent,
    uses an LLM to assign one or more categories from the Franzoi et al. (2025)
    taxonomy, and enriches the snippets with this classification data in-place.

    Args:
        state: The current graph state, which must contain `reranked_context_snippets`.

    Returns:
        An empty dictionary, as it modifies the state directly for efficiency.
    """
    logging.info("--- Running Franzoi Mapper Agent ---")

    # Step 1: Get the curated list of snippets from the state.
    context_snippets: List[ContextSnippet] = state.get("reranked_context_snippets", [])
    if not context_snippets:
        logging.warning("No context snippets found to classify.")
        return {} # Return empty dict as we are done
    
    # Log the source documents being processed for better traceability.
    # This log message shows exactly which documents made it through the re-ranker.
    doc_sources = [Path(s['source_document']).name for s in context_snippets]
    logging.info(f"Received {len(doc_sources)} snippets to classify: {doc_sources}")

    # Step 2: Initialize the LLM with structured output capabilities.
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        error_msg = "OPENAI_API_KEY not found in .env file."
        logging.error(error_msg)
        return {"error": error_msg}

    llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
    structured_llm = llm.with_structured_output(ClassificationList)

    # Step 3: Load the persistent cache to avoid redundant API calls.
    llm_cache = load_cache()
    cache_updated = False

    # Step 4: Loop through and classify each snippet, using the cache to
    # maximize performance and reduce costs.
    for snippet in context_snippets:
        logging.info(f"Classifying snippet from: {snippet['source_document']}")

        # Sanitize the snippet text to prevent prompt injection issues
        sanitized_snippet = snippet["snippet_text"].replace('"', '\\"')
        prompt = PROMPT_TEMPLATE.format(snippet_text=sanitized_snippet)

        # Check the cache before making an expensive API call.
        cache_key = get_cache_key(prompt, MODEL_NAME)
        
        response_data = llm_cache.get(cache_key)
        if not response_data:
            logging.info(f"CACHE MISS. Calling API for snippet from: {snippet['source_document']}")
            try:
                response_object = structured_llm.invoke(prompt)
                response_data = response_object.dict()
                llm_cache[cache_key] = response_data
                cache_updated = True
            except Exception as e:
                logging.error(f"Error classifying snippet: {e}")
                snippet['classifications'] = [{"full_path": "CLASSIFICATION_FAILED", "reasoning": str(e)}]
                continue
        else:
            logging.info(f"CACHE HIT for snippet from: {snippet['source_document']}")

        # Step 5: Process the response and enrich the snippet object in-place.
        # This is an efficient way to add data without creating a new list.
        typed_classifications: List[FranzoiClassification] = [
            {"full_path": item.get("full_path", "UNKNOWN"), "reasoning": item.get("reasoning", "")}
            for item in response_data.get("classifications", [])
        ]
        snippet['classifications'] = typed_classifications
        logging.info(f"  > Classified as: {[c['full_path'] for c in typed_classifications]}")
        
    # If the cache was modified, save it back to the file.
    if cache_updated:
        save_to_cache(llm_cache)
        logging.info("LLM cache file updated.")


    logging.info("Franzoi Mapper Agent execution successful.")

    # This agent modifies the state in-place, so it returns an empty dictionary.
    return {}