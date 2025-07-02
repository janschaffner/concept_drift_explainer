import sys
import os
import json
from pathlib import Path
from datetime import datetime
import streamlit as st

# --- Path Correction ---
# This is crucial to allow Streamlit to find your backend modules
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
# -----------------------

from backend.graph.build_graph import build_graph

# --- Page Configuration ---
st.set_page_config(
    page_title="Concept Drift Explainer",
    page_icon="🤖",
    layout="wide"
)

# --- Main Application ---
st.title("🤖 Concept Drift Explanation Prototype")
st.markdown("""
Welcome to the Concept Drift Explainer, a prototype developed for a Design Science Research (DSR) master's thesis.
This tool leverages a multi-agent system built with LangGraph to explain *why* a concept drift occurred in a business process.

**How it works:**
1.  It starts with pre-detected drift information.
2.  It retrieves relevant documents from a knowledge base.
3.  It classifies the context using the Franzoi et al. (2025) taxonomy.
4.  Finally, it synthesizes all information into a human-readable explanation.

Press the button in the sidebar to begin the analysis.
""")

# --- Sidebar for Controls ---
with st.sidebar:
    st.header("Controls")
    # The main button that triggers the entire agent pipeline
    run_analysis = st.button("Run Drift Explanation Analysis")
    if run_analysis:
        # When a new analysis is run, reset the feedback state
        st.session_state.feedback_given = {}

# --- Session State Initialization ---
# We use session_state to store results so they persist across reruns
if 'explanation_result' not in st.session_state:
    st.session_state.explanation_result = None
if 'error_message' not in st.session_state:
    st.session_state.error_message = None
if 'feedback_given' not in st.session_state:
    st.session_state.feedback_given = {}

# --- Logic to Run the Graph ---
if run_analysis:
    # Show a spinner while the backend is working
    with st.spinner("🧠 Running full analysis pipeline... This may take a minute."):
        try:
            # Build and run the LangGraph application
            app = build_graph()
            initial_input = {}
            final_state = app.invoke(initial_input)

            # Store the results in the session state
            st.session_state.explanation_result = final_state.get('explanation')
            st.session_state.error_message = final_state.get('error')
            st.session_state.full_state = final_state # Store the full state for logging

        except Exception as e:
            st.session_state.error_message = f"An unexpected error occurred: {e}"
            st.session_state.explanation_result = None

# --- Displaying the Results ---
st.divider()
st.header("Analysis Results")

# Display an error if one occurred
if st.session_state.error_message:
    st.error(f"An error occurred during analysis: {st.session_state.error_message}")

# Display the explanation if it exists
elif st.session_state.explanation_result:
    explanation = st.session_state.explanation_result

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

                # --- Granular Feedback Mechanism ---
                st.markdown("---")
                
                # Check if feedback has been given for this specific cause
                if st.session_state.feedback_given.get(i):
                    st.success("Thank you for your feedback on this cause!")
                else:
                    st.write("Was this specific cause helpful?")
                    col1, col2, _ = st.columns([1, 1, 8])
                    
                    with col1:
                        if st.button("👍", key=f"up_{i}"):
                            st.session_state.feedback_given[i] = "positive"
                            
                            feedback_data = {
                                "timestamp": datetime.now().isoformat(),
                                "feedback_type": "positive",
                                "cause_rated": cause,
                                "full_state": st.session_state.get('full_state', {})
                            }
                            # Define the path for the feedback log
                            feedback_dir = project_root / "data" / "feedback"
                            feedback_dir.mkdir(exist_ok=True)
                            feedback_file = feedback_dir / "feedback_log.jsonl"
                            # Append feedback as a new line in a JSONL file
                            with open(feedback_file, "a") as f:
                                f.write(json.dumps(feedback_data) + "\n")
                            
                            print(f"Positive feedback for cause #{i} logged to {feedback_file}")
                            st.rerun()

                    with col2:
                        if st.button("👎", key=f"down_{i}"):
                            st.session_state.feedback_given[i] = "negative"

                            feedback_data = {
                                "timestamp": datetime.now().isoformat(),
                                "feedback_type": "negative",
                                "cause_rated": cause,
                                "full_state": st.session_state.get('full_state', {})
                            }
                            feedback_dir = project_root / "data" / "feedback"
                            feedback_dir.mkdir(exist_ok=True)
                            feedback_file = feedback_dir / "feedback_log.jsonl"
                            with open(feedback_file, "a") as f:
                                f.write(json.dumps(feedback_data) + "\n")

                            print(f"Negative feedback for cause #{i} logged to {feedback_file}")
                            st.rerun()
else:
    st.info("Click the 'Run Analysis' button in the sidebar to start.")