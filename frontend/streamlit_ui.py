import sys
import os
from pathlib import Path
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

# --- Session State Initialization ---
# We use session_state to store the result so it persists across reruns
if 'explanation_result' not in st.session_state:
    st.session_state.explanation_result = None
if 'error_message' not in st.session_state:
    st.session_state.error_message = None

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
            # Use an expander for each cause to keep the UI clean
            with st.expander(f"**Cause #{i+1}:** {cause.get('context_category', 'N/A')}", expanded=i==0):
                st.markdown(f"**Description:** {cause.get('cause_description', 'N/A')}")
                st.markdown(f"**Confidence:** `{cause.get('confidence_score', 'N/A')*100:.1f}%`")
                
                st.markdown("---")
                
                # Use a code block to display the evidence snippet
                st.markdown("**Evidence:**")
                st.code(cause.get('evidence_snippet', 'N/A'), language='text')
                st.caption(f"Source: `{cause.get('source_document', 'N/A')}`")

else:
    st.info("Click the 'Run Analysis' button in the sidebar to start.")