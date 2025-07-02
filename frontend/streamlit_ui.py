# frontend/streamlit_ui.py

import sys
import json
from pathlib import Path
from datetime import datetime
import streamlit as st

# --- Path Correction ---
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
# -----------------------

from backend.graph.build_graph import build_graph
from backend.agents.chatbot_agent import run_chatbot_agent

# --- Page Configuration ---
st.set_page_config(
    page_title="Concept Drift Explainer",
    page_icon="🤖",
    layout="wide"
)

# --- Session State Initialization ---
def init_session_state():
    """Initializes all necessary session state variables."""
    if 'explanation_result' not in st.session_state:
        st.session_state.explanation_result = None
    if 'error_message' not in st.session_state:
        st.session_state.error_message = None
    if 'feedback_states' not in st.session_state:
        st.session_state.feedback_states = {}
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'full_state' not in st.session_state:
        st.session_state.full_state = {}

# --- Main Application ---
init_session_state()

st.title("🤖 Concept Drift Explanation Prototype")
st.markdown("Welcome! Press the button in the sidebar to begin the analysis.")

# --- Sidebar for Controls ---
with st.sidebar:
    st.header("Controls")
    if st.button("Run Drift Explanation Analysis"):
        init_session_state()
        with st.spinner("🧠 Running full analysis pipeline... This may take a minute."):
            try:
                app = build_graph()
                initial_input = {}
                final_state = app.invoke(initial_input)
                st.session_state.full_state = final_state
                st.session_state.explanation_result = final_state.get('explanation')
                st.session_state.error_message = final_state.get('error')
            except Exception as e:
                st.session_state.error_message = f"An unexpected error occurred: {e}"
                st.session_state.explanation_result = None
        st.rerun()

# --- Chat Dialog Logic ---
@st.dialog("Conversational Analysis")
def run_chat_dialog():
    """Renders the chat interface inside a dialog window."""
    st.write("Ask follow-up questions about the generated explanation.")
    
    # Display past messages from session state
    for author, message in st.session_state.chat_history:
        with st.chat_message(author):
            st.markdown(message)

    # The main chat input box
    if user_question := st.chat_input("Ask your question..."):
        # Add user's message to history and display it
        st.session_state.chat_history.append(("user", user_question))
        with st.chat_message("user"):
            st.markdown(user_question)

        # Prepare state and call the chatbot agent
        current_state = st.session_state.full_state
        current_state['user_question'] = user_question
        current_state['chat_history'] = st.session_state.chat_history
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = run_chatbot_agent(current_state)
                if response.get("error"):
                    st.error(f"Error from chatbot: {response['error']}")
                else:
                    # Update history with the response and display AI message
                    st.session_state.chat_history = response.get('chat_history', [])
                    ai_answer = st.session_state.chat_history[-1][1]
                    st.markdown(ai_answer)


# --- Displaying the Main Results ---
st.divider()
st.header("Analysis Results")

if st.session_state.error_message:
    st.error(f"An error occurred during analysis: {st.session_state.error_message}")
elif st.session_state.explanation_result:
    explanation = st.session_state.explanation_result

    # Display Summary and Ranked Causes
    st.subheader("Executive Summary")
    st.info(explanation.get('summary', 'No summary available.'))
    st.subheader("Ranked Potential Causes")
    
    ranked_causes = explanation.get('ranked_causes', [])
    if not ranked_causes:
        st.warning("No specific causes were identified.")
    else:
        for i, cause in enumerate(ranked_causes):
            with st.expander(f"**Cause #{i+1}:** {cause.get('context_category', 'N/A')}", expanded=i==0):
                st.markdown(f"**Description:** {cause.get('cause_description', 'N/A')}")
                st.markdown(f"**Confidence:** `{cause.get('confidence_score', 0.0)*100:.1f}%`")
                st.markdown("---")
                st.markdown("**Evidence:**")
                st.code(cause.get('evidence_snippet', 'N/A'), language='text')
                st.caption(f"Source: `{cause.get('source_document', 'N/A')}`")

                # Granular Feedback Mechanism
                st.markdown("---")
                if st.session_state.feedback_states.get(i):
                    st.success("Thank you for your feedback on this cause!")
                else:
                    st.write("Was this specific cause helpful?")
                    col1, col2, _ = st.columns([1, 1, 8])
                    with col1:
                        if st.button("👍", key=f"up_{i}"):
                            st.session_state.feedback_states[i] = "positive"
                            feedback_data = {"timestamp": datetime.now().isoformat(), "feedback_type": "positive", "cause_rated": cause, "full_state": st.session_state.get('full_state', {})}
                            feedback_dir = project_root / "data" / "feedback"
                            feedback_dir.mkdir(exist_ok=True)
                            feedback_file = feedback_dir / "feedback_log.jsonl"
                            with open(feedback_file, "a") as f: f.write(json.dumps(feedback_data) + "\n")
                            print(f"Positive feedback for cause #{i} logged to {feedback_file}")
                            st.rerun()
                    with col2:
                        if st.button("👎", key=f"down_{i}"):
                            st.session_state.feedback_states[i] = "negative"
                            feedback_data = {"timestamp": datetime.now().isoformat(), "feedback_type": "negative", "cause_rated": cause, "full_state": st.session_state.get('full_state', {})}
                            feedback_dir = project_root / "data" / "feedback"
                            feedback_dir.mkdir(exist_ok=True)
                            feedback_file = feedback_dir / "feedback_log.jsonl"
                            with open(feedback_file, "a") as f: f.write(json.dumps(feedback_data) + "\n")
                            print(f"Negative feedback for cause #{i} logged to {feedback_file}")
                            st.rerun()

    # Button to launch Chat
    st.divider()
    if st.button("💬 Ask Follow-up Questions"):
        # Reset chat history when opening a new dialog
        st.session_state.chat_history = []
        run_chat_dialog()

else:
    st.info("Click the 'Run Analysis' button in the sidebar to start.")