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

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    Classifies retrieved context snippets using the Franzoi taxonomy via LLM tool-calling.
    """
    logging.info("--- Running Franzoi Mapper Agent (Tool-Calling Version) ---")
    
    retrieved_snippets: List[ContextSnippet] = state.get("raw_context_snippets", [])
    if not retrieved_snippets:
        logging.warning("No context snippets found to classify.")
        return {"classified_context": []}

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        error_msg = "OPENAI_API_KEY not found in .env file."
        logging.error(error_msg)
        return {"error": error_msg}
        
    # Initialize the LLM
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    # This is the new, more reliable method for getting structured output.
    # We bind our Pydantic schema (ClassificationList) to the LLM as a "tool".
    structured_llm = llm.with_structured_output(ClassificationList)
    
    classified_snippets: List[ContextSnippet] = []
    for snippet in retrieved_snippets:
        logging.info(f"Classifying snippet from: {snippet['source_document']}")
        
        prompt = PROMPT_TEMPLATE.format(snippet_text=snippet["snippet_text"])
        
        try:
            # The LLM is now forced to return an object matching the ClassificationList schema.
            response_object = structured_llm.invoke(prompt)
            
            # The output is already a Pydantic object, not a JSON string.
            # We convert it to a list of dicts for our state.
            typed_classifications: List[FranzoiClassification] = [
                {"full_path": item.full_path, "reasoning": item.reasoning}
                for item in response_object.classifications
            ]
            
            snippet['classifications'] = typed_classifications
            logging.info(f"  > Classified as: {[c['full_path'] for c in typed_classifications]}")

        except Exception as e:
            logging.error(f"Error classifying snippet: {e}")
            snippet['classifications'] = [{"full_path": "CLASSIFICATION_FAILED", "reasoning": str(e)}]
        
        classified_snippets.append(snippet)

    logging.info("Franzoi Mapper Agent execution successful.")
    
    return {"classified_context": classified_snippets}