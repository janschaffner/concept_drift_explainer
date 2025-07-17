import sys
import pandas as pd
import json
import ast
from pathlib import Path
from datetime import datetime
import streamlit as st
import warnings

# --- Path Correction ---
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
# -----------------------

# Suppress the specific Pydantic V1 deprecation warning from LangChain
warnings.filterwarnings("ignore", message=".*Pydantic BaseModel V1.*")

from backend.graph.build_graph import build_graph
from backend.agents.chatbot_agent import run_chatbot_agent
from backend.agents.drift_linker_agent import run_drift_linker_agent
from backend.utils.ingest_documents import process_context_files
from backend.agents.drift_linker_agent import ConnectionType

# --- Descriptions for Drift Linker Categories ---
CONNECTION_TYPE_DESCRIPTIONS = {
    ConnectionType.STRONG_CAUSAL.value: "One drift appears to be a direct cause of another subsequent drift.",
    ConnectionType.SHARED_EVIDENCE.value: "The drifts are linked by common evidence or appear to share the same underlying root cause.",
    ConnectionType.THEMATIC_OVERLAP.value: "The drifts are not directly linked but share a similar theme or occur in related business areas.",
}

# --- UI Helper Function ---
def get_date_from_filename(filename: str) -> str:
    """Parses a YYYY-MM-DD date from the start of a filename."""
    try:
        # Handle potential path objects
        filename = Path(filename).name
        date_str = filename.split('_')[0]
        # Parse and reformat the date
        dt_obj = datetime.strptime(date_str, '%Y-%m-%d')
        return dt_obj.strftime('%d.%m.%Y')
    except (ValueError, IndexError):
        return "N/A"

# --- Helper Function to Load and Unpack Drifts ---
@st.cache_data
def load_and_unpack_drifts():
    """
    Loads drift data and unpacks rows with multiple drifts into a list.
    """
    csv_path = project_root / "data" / "drift_outputs" / "prediction_results.csv"
    try:
        df = pd.read_csv(csv_path)
        
        drift_options = []
        # This logic handles files where one row might contain multiple drifts in its columns
        for row_index, row in df.iterrows():
            drift_types = ast.literal_eval(row['Detected Drift Types'])
            changepoints = ast.literal_eval(row['Detected Changepoints'])
            
            for drift_index, drift_type in enumerate(drift_types):
                option_id = f"{row_index}-{drift_index}"
                display_name = (
                    f"Drift #{len(drift_options) + 1}: {drift_type} "
                    f"(between {changepoints[drift_index][0]} and {changepoints[drift_index][1]})"
                )
                drift_options.append({"id": option_id, "display": display_name})
        
        return drift_options
        
    except FileNotFoundError:
        st.error(f"Could not find drift data at {csv_path}")
        return []
    except Exception as e:
        st.error(f"Error loading or parsing drift data: {e}")
        return []

# --- Page Configuration ---
st.set_page_config(page_title="Concept Drift Explainer", page_icon="🤖", layout="wide")

# --- Session State Initialization ---
def init_session_state():
    """Initializes all necessary session state variables for a new run."""
    st.session_state.all_explanations = []
    st.session_state.error_message = None
    st.session_state.feedback_states = {}
    st.session_state.chat_history = []
    st.session_state.full_state_log = []
    st.session_state.linked_drift_summary = None
    st.session_state.connection_type = None

# --- Main Application ---
# Initialize state on first load if it doesn't exist
if 'all_explanations' not in st.session_state:
    init_session_state()

st.title("🤖 Concept Drift Explainer")
st.markdown("Welcome! Press the button in the sidebar to begin the analysis of all detected drifts.")

# --- Hallucination Warning ---
st.warning(
    "The Concept Drift Explainer can make mistakes. Check important information.", 
    icon="⚠️"
)

drift_options = load_and_unpack_drifts()
DOCUMENTS_PATH = project_root / "data" / "documents"

# --- Sidebar for Controls ---
with st.sidebar:
    st.header("Controls")
    
    # --- File Uploader Section ---
    st.divider()
    st.header("Add New Context")
    uploaded_files = st.file_uploader(
        "Upload new documents (.pdf, .pptx, etc.)",
        accept_multiple_files=True,
        type=['pdf', 'pptx', 'docx', 'txt', 'png', 'jpg']
    )

    if st.button("Process Uploaded Files") and uploaded_files:
        saved_files_paths = []
        with st.spinner("Saving and processing uploaded files..."):
            for uploaded_file in uploaded_files:
                # We must save the file to our documents folder first
                file_path = DOCUMENTS_PATH / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                saved_files_paths.append(file_path)
            
            # Now, call the ingestion logic with only the new files
            process_context_files(saved_files_paths)
            st.success(f"Successfully processed {len(saved_files_paths)} new document(s)!")
            # Clear the drift data cache so the UI reloads it if necessary
            st.cache_data.clear()
            st.rerun()

    st.divider()
    st.header("Run Analysis")

    if drift_options:
        st.info(f"Found **{len(drift_options)}** drift(s) to analyze.")
        
        if st.button("Run Full Analysis"):
            init_session_state() # Reset state for a new run
            all_explanations = []
            full_state_log = []
            
            progress_bar = st.progress(0.0, text="Starting Analysis...")
            
            with st.spinner(f"Analyzing {len(drift_options)} drift(s)...This may take several minutes."):
                try:
                    app = build_graph()
                    # Loop to process each drift sequentially
                    for i, drift in enumerate(drift_options):
                        progress = (i + 1) / len(drift_options)
                        progress_bar.progress(progress, text=f"Analyzing {drift['display']}...")
                        
                        row_idx, drift_idx = map(int, drift['id'].split('-'))
                        initial_input = {"selected_drift": {"row_index": row_idx, "drift_index": drift_idx}}
                        final_state = app.invoke(initial_input)
                        
                        if final_state.get("error"):
                            st.session_state.error_message = f"Error on {drift['display']}: {final_state['error']}"
                            break 
                        
                        st.toast(f"✅ {drift['display']} successfully analyzed!")
                        
                        all_explanations.append(final_state.get('explanation'))
                        full_state_log.append(final_state)

                    if not st.session_state.error_message and len(all_explanations) > 1:
                        st.toast("🔗 Analyzing relationships between drifts...")
                        linker_result = run_drift_linker_agent(full_state_log) # Use the local variable with current results
                        st.session_state.linked_drift_summary = linker_result.get("linked_drift_summary")
                        st.session_state.connection_type = linker_result.get("connection_type")

                    if not st.session_state.error_message:
                        st.session_state.all_explanations = all_explanations
                        st.session_state.full_state_log = full_state_log

                except Exception as e:
                    st.session_state.error_message = f"An unexpected error occurred: {e}"
            
            progress_bar.empty()
            st.rerun()
    else:
        st.error("Could not find or parse `prediction_results.csv`.")

# --- Chat Dialog Logic ---
@st.dialog("Conversational Analysis")
def run_chat_dialog():
    st.write("Ask follow-up questions about the generated explanations.")
    
    for author, message in st.session_state.chat_history:
        with st.chat_message(author):
            st.markdown(message)

    if user_question := st.chat_input("Ask your question..."):
        st.session_state.chat_history.append(("user", user_question))
        with st.chat_message("user"):
            st.markdown(user_question)

        # For the chatbot, we can provide the full log of all states as context
        current_state = {"full_state_log": st.session_state.full_state_log}
        current_state['user_question'] = user_question
        current_state['chat_history'] = st.session_state.chat_history
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = run_chatbot_agent(current_state)
                if response.get("error"):
                    st.error(f"Error from chatbot: {response['error']}")
                else:
                    st.session_state.chat_history = response.get('chat_history', [])
                    ai_answer = st.session_state.chat_history[-1][1]
                    st.markdown(ai_answer)

# --- Displaying the Main Results ---
st.divider()
st.header("Analysis Results")

if st.session_state.error_message:
    st.error(f"An error occurred during analysis: {st.session_state.error_message}")

elif st.session_state.all_explanations:
    st.success(f"Successfully analyzed {len(st.session_state.all_explanations)} drift(s).")
    
    # --- Display the linked drift analysis summary ---
    if st.session_state.linked_drift_summary:
        connection_type = st.session_state.connection_type
        description = CONNECTION_TYPE_DESCRIPTIONS.get(connection_type, "A potential link was identified between the drifts.")
        with st.container(border=True):
            st.subheader("🔗 Cross-Drift Analysis")
            st.markdown(f"**Connection Type:** {connection_type}")
            st.caption(description)
            st.markdown("---")
            st.markdown(st.session_state.linked_drift_summary)
    
    # Loop through each explanation object in the list
    for explanation_idx, explanation in enumerate(st.session_state.all_explanations):
        # Dynamically construct a more informative title
        drift_state = st.session_state.full_state_log[explanation_idx]
        drift_info = drift_state.get("drift_info", {})
        drift_type = drift_info.get("drift_type", "Unknown").capitalize()
        # Parse and reformat dates for the title from YYYY-MM-DD to DD.MM.YYYY
        try:
            start_date_str = drift_info.get("start_timestamp", "N/A").split(" ")[0]
            end_date_str = drift_info.get("end_timestamp", "N/A").split(" ")[0]
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d').strftime('%d.%m.%Y')
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d').strftime('%d.%m.%Y')
        except (ValueError, IndexError):
            start_date, end_date = "N/A", "N/A"
        # Create a two-line header for clarity
        st.subheader(f"Explanation for Drift #{explanation_idx + 1}: {drift_type}")
        st.markdown(f"<h4 style='margin-top: -0.75rem; '>Timeframe: {start_date} – {end_date}</h4>", unsafe_allow_html=True)
        changepoints = drift_info.get("changepoints", ("N/A", "N/A"))
        activity_str = f"{changepoints[0]} → {changepoints[1]}"
        st.write(f"##### Activities: {activity_str}")

        st.info(explanation.get('summary', 'No summary available.'))
        
        ranked_causes = explanation.get('ranked_causes', [])
        if not ranked_causes:
            st.warning("No specific causes were identified for this drift.")
        else:
            # Loop through each cause within an explanation
            for cause_idx, cause in enumerate(ranked_causes):
                # Add a one-line guard-rail that prevents that glossary entries are displayed
                if "bpm glossary" in cause.get("source_document", "").lower():
                    continue   # hide glossary citations

                # Structured Cause Layout
                with st.container(border=True):
                    trigger_date = get_date_from_filename(cause.get("source_document", ""))
                    st.metric(label="**Drift Trigger Date**", value=trigger_date, help="The date parsed from the source document filename, indicating a potential trigger for the drift.")
                    st.markdown(f"**Confidence:** `{cause.get('confidence_score', 0.0)*100:.1f}%`")
                    st.markdown(f"**Source Document:** `{cause.get('source_document', 'N/A')}`")
                    
                    expander_title = f"**Cause #{cause_idx+1}:** {cause.get('context_category', 'N/A')}"
                    with st.expander(expander_title, expanded=False):
                        st.markdown(f"**Description:** {cause.get('cause_description', 'N/A')}")
                        st.markdown(f"**Confidence:** `{cause.get('confidence_score', 0.0)*100:.1f}%`")
                        #st.markdown("---")
                        st.markdown("**Evidence:**")
                        st.code(cause.get('evidence_snippet', 'N/A'), language='text')
                        st.caption(f"Source Document: `{cause.get('source_document', 'N/A')}`")

                    # Granular Feedback Mechanism
                    st.markdown("---")
                    feedback_key = (explanation_idx, cause_idx)
                    if st.session_state.feedback_states.get(feedback_key):
                        st.success("Thank you for your feedback on this cause!")
                    else:
                        st.write("Was this specific cause helpful?")
                        col1, col2, _ = st.columns([1, 1, 8])
                        with col1:
                            # Use both explanation_idx and cause_idx for a unique key
                            if st.button("👍", key=f"up_{explanation_idx}_{cause_idx}"):
                                st.session_state.feedback_states[feedback_key] = "positive"
                                feedback_data = {"timestamp": datetime.now().isoformat(), "feedback_type": "positive", "cause_rated": cause}
                                feedback_dir = project_root / "data" / "feedback"
                                feedback_dir.mkdir(exist_ok=True)
                                feedback_file = feedback_dir / "feedback_log.jsonl"
                                with open(feedback_file, "a") as f: f.write(json.dumps(feedback_data) + "\n")
                                print(f"Positive feedback logged for cause #{cause_idx+1} of drift #{explanation_idx+1}")
                                st.rerun()
                        with col2:
                            if st.button("👎", key=f"down_{explanation_idx}_{cause_idx}"):
                                st.session_state.feedback_states[feedback_key] = "negative"
                                feedback_data = {"timestamp": datetime.now().isoformat(), "feedback_type": "negative", "cause_rated": cause}
                                feedback_dir = project_root / "data" / "feedback"
                                feedback_dir.mkdir(exist_ok=True)
                                feedback_file = feedback_dir / "feedback_log.jsonl"
                                with open(feedback_file, "a") as f: f.write(json.dumps(feedback_data) + "\n")
                                print(f"Negative feedback logged for cause #{cause_idx+1} of drift #{explanation_idx+1}")
                                st.rerun()
        st.divider()

    # Button to launch Chat
    if st.button("💬 Ask Follow-up Questions about this Analysis"):
        st.session_state.chat_history = []
        run_chat_dialog()

else:
    st.info("Click 'Run Full Analysis' in the sidebar to start.")