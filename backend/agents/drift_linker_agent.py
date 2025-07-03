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

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    
    prompt_template = """You are a senior business process analyst conducting a meta-analysis. Your goal is to find high-level insights by identifying relationships between several independently explained concept drifts.

You will be provided with a list of explanations for multiple concept drifts that occurred in the same process log.

**## Explained Drifts**
{formatted_explanations}

**## Your Task**
Review all the provided drift explanations. Identify any potential relationships, common root causes, or cascading effects between them. For example:
- Are multiple drifts caused by the same source document (e.g., a single policy change causing drifts in different parts of the process)?
- Do the explanations share common keywords, themes, or context categories?
- Does one drift seem to be a direct consequence of another?

Summarize your findings in a short, insightful markdown-formatted paragraph. Start with a clear headline. If no clear relationships are found, state that explicitly.
"""
    
    prompt = prompt_template.format(formatted_explanations=formatted_explanations)
    
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        response = llm.invoke(prompt)
        linked_summary = response.content

        logging.info("Successfully generated linked drift summary.")
        return {"linked_drift_summary": linked_summary}

    except Exception as e:
        logging.error(f"Failed to generate linked drift summary: {e}")
        return {"error": str(e)}