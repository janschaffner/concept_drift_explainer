"""
This module contains the implementation of the Chatbot Agent.

Its primary responsibility is to provide an interactive, conversational Q&A layer
on top of the completed drift analysis. It is designed to be a stateful assistant
that can answer follow-up questions from the user, using the full context of all
previously analyzed drifts. It includes a topical guardrail to ensure the
conversation remains focused on the analysis.
"""

import os
import sys
import logging
from pathlib import Path
import textwrap
from typing import List, Dict

# --- Path Correction ---
# Ensures that the script can correctly import modules from the 'backend' directory.
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
# -----------------------

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic.v1 import BaseModel, Field

from backend.state.schema import GraphState
from backend.utils.cache import load_cache, save_to_cache, get_cache_key

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
MODEL_NAME = "gpt-4o-mini"

# --- Initialization ---
# Load environment variables and check for API key once at the module level
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY not found in environment variables.")

# --- Pydantic Models for Guardrail ---
class ChatbotGuardrail(BaseModel):
    """Boolean check for whether a user's question is on-topic."""
    is_on_topic: bool = Field(description="True if the user's question is relevant to the provided context, False otherwise.")

# --- Helper Functions ---

def format_chat_history(chat_history: list) -> str:
    """Formats the chat history into a string for the LLM prompt.

    Args:
        chat_history: A list of tuples, where each tuple is a (user, assistant)
                      exchange.

    Returns:
        A formatted string representing the conversation history.
    """
    if not chat_history:
        return "No previous conversation."
    return "\n".join([f"Human: {q}\nAssistant: {a}" for q, a in chat_history])

def format_full_analysis_context(full_state_log: List[Dict]) -> str:
    """Formats the entire analysis log into a compact context block for the LLM.

    This function iterates through the log of all completed drift analyses,
    extracting the key findings (summary, causal documents, etc.) for each one.
    This provides the chatbot with a comprehensive, at-a-glance summary of all
    the information it needs to answer user questions.

    Args:
        full_state_log: A list of GraphState dictionaries, one for each
                        completed drift analysis.

    Returns:
        A formatted string summarizing all drift analyses.
    """
    context_str = ""
    for i, state in enumerate(full_state_log, 1):
        drift_info = state.get('drift_info', {})
        explanation = state.get('explanation', {})
        ranked_causes = explanation.get('ranked_causes', [])
        
        causes_list = "\n".join(
            f"    - {c.get('source_document', 'N/A')}: \"{c.get('evidence_snippet', '')[:100]}…\""
            for c in ranked_causes
        ) or "    - None"

        context_str += textwrap.dedent(f"""\
        ### Drift #{i}: {drift_info.get('drift_type')}
        - **Timeframe:** {drift_info.get('start_timestamp')} to {drift_info.get('end_timestamp')}
        - **Summary:** {explanation.get('summary')}
        - **Causal Documents:**
        {causes_list}
        """)
    return context_str

def is_on_topic(user_question: str, context: str) -> bool:
    """Uses an LLM to check if a user's question is relevant to the analysis.

    This function acts as a "topical guardrail" to prevent the chatbot from
    engaging in off-topic conversations. It uses a few-shot prompt to give the
    LLM clear examples of what is and isn't on-topic, improving its accuracy.

    Args:
        user_question: The question asked by the user.
        context: The full analysis context.

    Returns:
        True if the question is on-topic, False otherwise.
    """
    logging.info("--- Running Topical Guardrail Check ---")
    
    prompt_template = textwrap.dedent("""\
    You are a topic-classification assistant. Your task is to determine if a user's question is relevant to the provided 'Analysis Context'.

    An on-topic question is one that asks about the concept drift, the business process, the evidence documents, or seeks clarification on the content within the analysis.
    An off-topic question asks about something completely unrelated to the provided context.

    **Analysis Context:**
    {context}

    ---
    Here are some examples:

    - User Question: "Who was responsible for the new pre-approval step?"
    - Your Decision: On-topic. (This is a valid question about the content of the evidence).

    - User Question: "Can you summarize the 'Compliance Guidance' document?"
    - Your Decision: On-topic. (This is a valid question about one of the evidence documents).

    - User Question: "What is the recipe for a strawberry cake?"
    - Your Decision: Off-topic. (This is completely unrelated to the analysis).
    ---

    Now, based on the provided 'Analysis Context', make a decision for the following question:

    **User's Question:**
    {question}

    Is the user's question on-topic?
    """)
    
    prompt = prompt_template.format(context=context, question=user_question)
    
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
    structured_llm = llm.with_structured_output(ChatbotGuardrail)
    
    try:
        response = structured_llm.invoke(prompt)
        logging.info(f"Guardrail check complete. On-topic: {response.is_on_topic}")
        return response.is_on_topic
    except Exception as e:
        logging.error(f"Error in topical guardrail check: {e}")
        # Default to assuming the question is on-topic in case of an error
        return True
    
# --- Main Agent Logic ---

def run_chatbot_agent(state: GraphState) -> dict:
    """
    Answers a user's follow-up question based on the full analysis context
    and the ongoing conversation history.

    This agent is part of the graph's main loop and is only called if the user
    submits a question. It first uses a topical guardrail to validate the
    question's relevance before generating a contextualized answer.

    Args:
        state: The current graph state, which must contain the `user_question`.

    Returns:
        A dictionary with the updated `chat_history`.
    """
    logging.info("--- Running Chatbot Agent ---")
    
    user_question = state.get("user_question")
    if not user_question:
        return {"error": "No user question provided."}

    # Step 1: Assemble the full context from the analysis log and chat history.
    # The chatbot now receives the full log of all states.
    full_state_log = state.get('full_state_log', [])
    chat_history = state.get('chat_history', [])
    
    # Format the new, comprehensive context
    full_analysis_context = format_full_analysis_context(full_state_log)

    # Step 2: Apply the topical guardrail to ensure the question is relevant.
    # If it's not, return a polite refusal and do not proceed.
    if not is_on_topic(user_question, full_analysis_context):
        ai_answer = "I am an assistant for analyzing concept drifts. I can only answer questions related to the drift analysis, the process, and the provided evidence. How can I help you with the analysis?"
        new_history = chat_history + [(user_question, ai_answer)]
        return {"chat_history": new_history}
    
    # Step 3: If the guardrail passes, generate a contextualized answer using the LLM. 
    full_context = textwrap.dedent(f"""
        **Full Analysis Report:**
        {full_analysis_context}
        **Previous Conversation:**
        {format_chat_history(chat_history)}
        """)

    prompt_template = textwrap.dedent("""\
    You are a helpful AI assistant having a conversation with a business analyst. The analyst is asking follow-up questions about a concept drift explanation that you have already provided.
    Use the provided "Original Analysis Context" and "Previous Conversation" to answer the "User's New Question". Keep your answers concise and helpful. You can now refer to the "Causal Documents" by name in your answer.

    ---
    {context}
    ---

    **User's New Question:**
    {question}

    **Your Answer:**
    """)
    prompt = prompt_template.format(context=full_context, question=user_question)

    # Log the full prompt for debugging purposes.
    logging.info(f"Chatbot prompt created:\n{prompt}")

    # Use a non-zero temperature to make the chatbot's responses more creative.
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0.3)
    
    # Step 4: Use caching to improve performance and reduce costs for repeated questions.
    llm_cache = load_cache()
    cache_key = get_cache_key(prompt, MODEL_NAME)

    ai_answer = llm_cache.get(cache_key)
    if not ai_answer:
        logging.info(f"CACHE MISS. Calling API for user question: '{user_question}'")
        try:
            response = llm.invoke(prompt)
            ai_answer = response.content
            
            llm_cache[cache_key] = ai_answer
            save_to_cache(llm_cache)
            logging.info("Chatbot response cached successfully.")
        except Exception as e:
            logging.error(f"Error in Chatbot Agent: {e}")
            return {"error": str(e)}
    else:
        logging.info(f"CACHE HIT for user question: '{user_question}'")

    # Step 5: Append the new Q&A turn to the chat history and return the updated state.
    new_history = chat_history + [(user_question, ai_answer)]
    
    # Return the updated chat history to the state.
    return {"chat_history": new_history}