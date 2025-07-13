import os
import sys
import logging
from pathlib import Path

# --- Path Correction ---
# Ensures that the script can correctly import modules from the 'backend' directory.
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
# -----------------------

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from backend.state.schema import GraphState
from backend.utils.cache import load_cache, save_to_cache, get_cache_key

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
MODEL_NAME = "gpt-4o-mini"

def format_chat_history(chat_history: list) -> str:
    """
    Formats the chat history into a string for the LLM prompt.

    Args:
        chat_history: A list of tuples, where each tuple is a (user, assistant) exchange.

    Returns:
        A formatted string representing the conversation history.
    """
    if not chat_history:
        return "No previous conversation."
    return "\n".join([f"Human: {q}\nAssistant: {a}" for q, a in chat_history])

def run_chatbot_agent(state: GraphState) -> dict:
    """
    Answers a user's follow-up question based on the full analysis context
    and the ongoing conversation history.

    This agent is part of the graph's main loop and is only called if the user
    submits a question through the UI's chat dialog.

    Args:
        state: The current graph state, which must contain the `user_question`.

    Returns:
        A dictionary with the updated `chat_history`.
    """
    logging.info("--- Running Chatbot Agent ---")
    
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        return {"error": "OPENAI_API_KEY not found."}

    user_question = state.get("user_question")
    if not user_question:
        return {"error": "No user question provided."}

    # Prepare the context for the prompt
    # The chatbot receives the full state log, giving it access to all previous
    # agent outputs, including the drift info and the final explanation.
    drift_info = state.get('drift_info', {})
    explanation = state.get('explanation', {})
    chat_history = state.get('chat_history', [])

    # Construct a context string that includes the original analysis and the conversation so far.
    full_context = f"""
    **Original Analysis Context:**
    - Drift Detected: {drift_info.get('drift_type')} from {drift_info.get('start_timestamp')} to {drift_info.get('end_timestamp')}.
    - Explanation Summary: {explanation.get('summary')}

    **Previous Conversation:**
    {format_chat_history(chat_history)}
    """

    prompt_template = """You are a helpful AI assistant having a conversation with a business analyst. The analyst is asking follow-up questions about a concept drift explanation that you have already provided.

Use the provided "Original Analysis Context" and "Previous Conversation" to answer the "User's New Question". Keep your answers concise and helpful.

---
{context}
---

**User's New Question:**
{question}

**Your Answer:**
"""
    prompt = prompt_template.format(context=full_context, question=user_question)

    # Log the full prompt for debugging purposes.
    logging.info(f"Chatbot prompt created:\n{prompt}")

    # Use a non-zero temperature to make the chatbot's responses more creative.
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0.3)
    
    # Caching Logic
    llm_cache = load_cache()
    cache_key = get_cache_key(prompt, MODEL_NAME)

    if cache_key in llm_cache:
        logging.info(f"CACHE HIT for user question: '{user_question}'")
        ai_answer = llm_cache[cache_key]
    else:
        logging.info(f"CACHE MISS. Calling API for user question: '{user_question}'")
        try:
            response = llm.invoke(prompt)
            ai_answer = response.content
            
            # Save the new response to the cache.
            llm_cache[cache_key] = ai_answer
            save_to_cache(llm_cache)
            logging.info("Chatbot response cached successfully.")
        except Exception as e:
            logging.error(f"Error in Chatbot Agent: {e}")
            return {"error": str(e)}

    # Append the new interaction to the history.
    # The previous history is already in the state, so we just append the latest interaction.
    new_history = chat_history + [(user_question, ai_answer)]
    
    # Return the updated chat history to the state.
    return {"chat_history": new_history}