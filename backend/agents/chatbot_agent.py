import os
import sys
import logging
from pathlib import Path

# --- Path Correction ---
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
# -----------------------

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from backend.state.schema import GraphState

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def format_chat_history(chat_history: list) -> str:
    """Formats the chat history into a string for the prompt."""
    if not chat_history:
        return "No previous conversation."
    return "\n".join([f"Human: {q}\nAssistant: {a}" for q, a in chat_history])

def run_chatbot_agent(state: GraphState) -> dict:
    """
    Answers a user's follow-up question based on the full analysis context.
    """
    logging.info("--- Running Chatbot Agent ---")
    
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        return {"error": "OPENAI_API_KEY not found."}

    user_question = state.get("user_question")
    if not user_question:
        return {"error": "No user question provided."}

    # Prepare the context for the prompt
    drift_info = state.get('drift_info', {})
    explanation = state.get('explanation', {})
    chat_history = state.get('chat_history', [])

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

    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        response = llm.invoke(prompt)
        ai_answer = response.content

        # Append the new interaction to the history
        new_history = chat_history + [(user_question, ai_answer)]
        
        return {"chat_history": new_history}

    except Exception as e:
        logging.error(f"Error in Chatbot Agent: {e}")
        return {"error": str(e)}