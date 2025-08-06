import sys
from pathlib import Path
import warnings
import pandas as pd
import json
import ast
from datetime import datetime
import streamlit as st
from streamlit_timeline import timeline
from typing import Optional

# Path Correction
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

# Suppress LangChain Pydantic V1 deprecation warning
warnings.filterwarnings("ignore", message=".*Pydantic BaseModel V1.*")

# Backend Imports
from backend.graph.build_graph import build_graph
from backend.agents.chatbot_agent import run_chatbot_agent
from backend.agents.drift_linker_agent import run_drift_linker_agent, ConnectionType
from backend.utils.reporting import generate_docx_report

# Descriptions for Drift Linker Categories
CONNECTION_TYPE_DESCRIPTIONS = {
    ConnectionType.STRONG_CAUSAL.value:   "One drift appears to be a direct cause of another subsequent drift.",
    ConnectionType.SHARED_EVIDENCE.value: "The drifts are linked by common evidence or appear to share the same underlying root cause.",
    ConnectionType.THEMATIC_OVERLAP.value:"The drifts are not directly linked but share a similar theme or occur in related business areas.",
}

# --- HELPER FUNCTIONS SECTION ---

# Floating chat dialog (modal)
@st.dialog("Conversational Analysis")
def run_chat_dialog():
    if not st.session_state.chat_history:
        st.session_state.chat_history.append(
            ("assistant", "Hello! Ask me anything about the analysis."))
        
    for author, msg in st.session_state.chat_history:
        avatar = "assets/user_avatar.png" if author=="user" else "assets/chatbot_avatar.png"
        with st.chat_message(author, avatar=avatar):
            st.markdown(msg)

    if prompt := st.chat_input("Your question..."):
        st.session_state.chat_history.append(("user", prompt))
        with st.chat_message("assistant", avatar="assets/chatbot_avatar.png"):
            with st.spinner("Thinking..."):
                resp = run_chatbot_agent({
                    "full_state_log": st.session_state.full_state_log,
                    "chat_history": st.session_state.chat_history,
                    "user_question": prompt})
                st.session_state.chat_history = resp["chat_history"]
                st.markdown(st.session_state.chat_history[-1][1])
        st.rerun()

# Helper: Parse date out of filename
def get_datetime_from_filename(filename: str) -> Optional[datetime]:
    """
    Parses a date from a filename (e.g., '2025-07-30_...') and returns
    a datetime object. Returns None if parsing fails.
    """
    try:
        name = Path(filename).name
        date_str = name.split('_')[0]
        return datetime.strptime(date_str, "%Y-%m-%d")
    except (ValueError, IndexError):
        return None

# Helper: Load & unpack drifts from drift_outputs/*.csv
@st.cache_data
def load_and_unpack_drifts(log_name: str):
    """
    Loads drift data for a specific event log if available,
    otherwise loads the single CSV in data/drift_outputs.
    """
    drift_root = project_root / "data" / "event_logs"
    # 1) Check for a subfolder matching the selected log
    candidate = drift_root / log_name
    if candidate.is_dir():
        csv_files = list(candidate.glob("*.csv"))
    else:
        # 2) Fallback: look at top‐level CSVs
        csv_files = list(drift_root.glob("*.csv"))

    if not csv_files:
        return []

    df = pd.read_csv(csv_files[0])
    options = []
    for row_idx, row in df.iterrows():
        drift_types  = ast.literal_eval(row["Detected Drift Types"])
        changepoints = ast.literal_eval(row["Detected Changepoints"])
        for d_idx, dtype in enumerate(drift_types):
            opt_id  = f"{row_idx}-{d_idx}"
            display = (
                f"Drift #{len(options)+1}: {dtype} "
                f"({changepoints[d_idx][0]} → {changepoints[d_idx][1]})"
            )
            options.append({"id": opt_id, "display": display})
    return options

def create_detailed_timeline(drift_event: dict, predicted_docs: list):
    """
    Creates a detailed timeline using the component's default styles,
    with ranked headlines for each cause.
    """
    events = []
    try:
        # Add the drift period bar
        start_dt = datetime.fromisoformat(drift_event["start_timestamp"].replace('Z', ''))
        end_dt = datetime.fromisoformat(drift_event["end_timestamp"].replace('Z', ''))
        events.append({
            "start_date": {"year": start_dt.year, "month": start_dt.month, "day": start_dt.day},
            "end_date": {"year": end_dt.year, "month": end_dt.month, "day": end_dt.day},
            "text": {
                "headline": "Drift Period",
                "text": "The calculated duration of the concept drift."},
            "group": "Drift Period"
        })
    except (ValueError, KeyError):
        pass

    # Use enumerate to get the index (rank) for each document, starting from 1.
    for i, doc in enumerate(predicted_docs, start=1):
        if 'timestamp' in doc and doc.get('timestamp'):
            doc_dt = datetime.fromtimestamp(doc['timestamp'])
            events.append({
                "start_date": {"year": doc_dt.year, "month": doc_dt.month, "day": doc_dt.day},
                "text": {
                    # Use the index 'i' to create a ranked headline.
                    "headline": f"Potential Cause #{i}",
                    "text": f"Document: <strong>{doc['source_document']}</strong>"},
                "group": "Potential Cause"
            })

    # 3. Render the timeline
    if len(events) > 1:
        timeline_data = {
            "font": "sans-serif",
            "events": events
        }
        timeline(timeline_data, height=350)

DRIFT_TYPE_EXPLANATIONS = {
    "sudden": "Sudden Drift: An abrupt, clean switch from an old process to a new one. After the change point, the old process is no longer used.",
    "gradual": "Gradual Drift: A slow transition where the old and new processes run side-by-side for a period. The new process is gradually rolled out as the old one is phased out.",
    "incremental": "Incremental Drift: A series of small, continuous adjustments that add up to a major process change over time, often seen in agile environments.",
    "recurring": "Recurring Drift: A process change that happens cyclically or seasonally. A previously used process version reappears for a temporary period."
}

def reset_analysis_results():
    """Clears only the results of a previous analysis run."""
    st.session_state.all_explanations = []
    st.session_state.error_message = None
    st.session_state.feedback_states = {}
    st.session_state.chat_history = []
    st.session_state.full_state_log = []
    st.session_state.linked_drift_summary = None
    st.session_state.connection_type = None
    st.session_state.analysis_run_complete = False
    st.session_state.show_chat = False

# Page Title
col1, col2 = st.columns([2, 20], vertical_alignment="center")

with col1:
    st.markdown('<p style="font-size: 4px;">&nbsp;</p>', unsafe_allow_html=True)
    st.image("assets/tab_icon.png", width=64)
with col2:
    st.title("Concept Drift Explainer", anchor=False)
st.divider()

# Hallucination Warning
#with st.container(border=True):
st.warning(
    "The Concept Drift Explainer can make mistakes. Verify important information!",
    icon="⚠️"
)

# --- CSS MODIFICATION SECTION ---

# empty atm

# --- STREAMLIT SECTION ---

# Top Controls: Event‐Log Dropdown + Run Button
logs_folder = project_root / "data" / "event_logs"
log_files = sorted([p.name for p in logs_folder.iterdir() if p.is_dir()])
placeholder = "▶️ Choose an event log..."

# Initialize selected_log once, so session_state.selected_log always exists
if "selected_log" not in st.session_state:
    st.session_state.selected_log = placeholder

# Define the two columns first
col_left, col_right = st.columns(2, gap="large", border=True)

# Left Column Card
with col_left:
    selected_log = st.selectbox(
        "Select Event Log to Analyze",
        options=log_files,
        key="selected_log",
        index=0,  # ensures the placeholder is shown initially
    )
    # only load drifts if a real log is chosen
    if selected_log != placeholder:
        drift_options = load_and_unpack_drifts(selected_log)
        n_drifts = len(drift_options)
        # show banner only when a real log is selected
        st.success(f"{n_drifts} drift(s) found.")
    else:
        # no log chosen → zero drifts
        drift_options = []
        n_drifts = 0

# Right Column Card
with col_right:
    # Added a title and spacer for better visual balance
    st.write("Ready to analyze?")
    #st.markdown("&nbsp;") # Vertical spacer

    run_disabled = (n_drifts == 0) or (selected_log == placeholder)
    if st.button("▶️ Run Drift Analysis", disabled=run_disabled):
        # Reset state
        reset_analysis_results()
        all_expls = []
        full_log  = []

        # Run backend graph pipeline
        with st.container(border=True):
            progress = st.progress(0.0, text="Starting analysis…")
            with st.spinner(f"Analyzing {n_drifts} drift(s)... This may take a moment."):
                graph_app = build_graph()
                for i, drift in enumerate(drift_options):
                    frac = (i + 1) / n_drifts
                    progress.progress(frac, text=f"Analyzing {drift['display']}…")
                    row_idx, d_idx = map(int, drift["id"].split("-"))
                    inp = {"selected_drift": {"row_index": row_idx, "drift_index": d_idx}}
                    state = graph_app.invoke(inp)
                    if state.get("error"):
                        st.session_state.error_message = f"Error in {drift['display']}: {state['error']}"
                        break
                    st.toast(f"✅ {drift['display']} analyzed")
                    all_expls.append(state.get("explanation"))
                    full_log.append(state)
                
                # Cross‐drift linking if multiple
                if not st.session_state.error_message and len(all_expls) > 1:
                    st.toast("🔗 Linking drifts…")
                    linker = run_drift_linker_agent(full_log)
                    st.session_state.linked_drift_summary = linker.get("linked_drift_summary")
                    st.session_state.connection_type      = linker.get("connection_type")
                
                # Save into session
                st.session_state.all_explanations   = all_expls
                st.session_state.full_state_log     = full_log
                st.session_state.analysis_run_complete = True
            progress.empty()
        st.rerun()

# Once analysis is done: divider + tabs for each drift
#st.divider()

if st.session_state.error_message:
    st.error(f"An error occurred: {st.session_state.error_message}")

elif st.session_state.all_explanations:
    #with st.container(border=True):
    exps = st.session_state.all_explanations
    st.success(f"Successfully analyzed {len(exps)} drift(s).")

        # Askfollowup button
        #if st.button("💬 Ask Follow-up Questions"):
            #st.session_state.show_chat = True

    #if st.session_state.show_chat:
        #run_chat_dialog()

    # Cross‐drift summary
    with st.container(border=True):
        if st.session_state.linked_drift_summary:
            ctype = st.session_state.connection_type
            desc = CONNECTION_TYPE_DESCRIPTIONS.get(ctype, "")
            with st.container():
                st.subheader("🔗 Cross-Drift Analysis")
                st.markdown(f"**Type:** {ctype}")
                st.caption(desc)
                st.markdown(st.session_state.linked_drift_summary)

    # Tabs: one per drift explanation (default to first)
    labels = [f"Drift #{i+1}" for i in range(len(exps))]
    tabs = st.tabs(labels)

    for idx, tab in enumerate(tabs):
        with tab:
            # This main container keeps the card-like appearance for the whole tab
            with st.container(border=True):
                # 1. Get and Prepare Data for this Tab
                explanation = exps[idx]
                state = st.session_state.full_state_log[idx]
                info = state.get("drift_info", {})
                
                causes_with_timestamps = []
                for cause in explanation.get("ranked_causes", []):
                    dt_object = get_datetime_from_filename(cause.get("source_document", ""))
                    if dt_object:
                        cause['timestamp'] = int(dt_object.timestamp())
                    else:
                        cause['timestamp'] = None
                    causes_with_timestamps.append(cause)

                # 2. Create the Main Header (Full-Width)
                dtype = info.get("drift_type", "Unknown").capitalize()
                start = info.get("start_timestamp","N/A").split(" ")[0]
                end = info.get("end_timestamp","N/A").split(" ")[0]
                try:
                    s_fmt = datetime.strptime(start, "%Y-%m-%d").strftime("%d.%m.%Y")
                    e_fmt = datetime.strptime(end, "%Y-%m-%d").strftime("%d.%m.%Y")
                    timeframe_str = f"(Timeframe: {s_fmt} – {e_fmt})"
                except Exception:
                    timeframe_str = ""

                # Create columns for the header and the export button
                header_col, button_col = st.columns([4, 1])

                # Look up the explanation for the current drift type
                drift_key = dtype.lower()
                help_text = DRIFT_TYPE_EXPLANATIONS.get(drift_key, "No explanation available for this drift type.")

                with header_col:
                    st.header(f"Explanation for Drift #{idx+1}: {dtype}", anchor=False, help=help_text)
                    st.subheader(timeframe_str, anchor=False)

                with button_col:
                    # Add some vertical space to push the button down
                    st.write("")

                    # Generate the report content by calling the new backend function
                    docx_bytes = generate_docx_report(info, explanation, drift_index=idx+1)

                    st.download_button(
                        label="📄 Export as DOCX",
                        data=docx_bytes,
                        file_name=f"drift_report_{idx+1}_{dtype.lower()}.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        help="Download the analysis for this drift as a DOCX file.",
                        use_container_width=True # Makes the button fill the column width
                    )

                # 3. Create the Two-Column Layout with Subheaders
                col_text, col_timeline = st.columns(2, gap="small", border=True)

                # Left Column: Textual Summary
                with col_text:
                    st.subheader("📝 Executive Summary", anchor=False)
                    st.info(explanation.get("summary", "No summary available."))

                # Right Column: Detailed Timeline
                with col_timeline:
                    st.subheader("🗓️ Causal Event Timeline", anchor=False)
                    create_detailed_timeline(
                        drift_event=info,
                        predicted_docs=causes_with_timestamps
                    )

            with st.container(border=True):
                # Get settings from session state
                max_causes_to_show = st.session_state.max_causes
                min_confidence = st.session_state.confidence_threshold

                # First, filter the list by the confidence threshold
                filtered_causes = [
                    cause for cause in causes_with_timestamps
                    if cause.get('confidence_score', 0) >= min_confidence
                ]

                # Calculate the actual number of causes to display
                actual_causes_shown = min(len(filtered_causes), max_causes_to_show)

                # Use the new, correct variable in the subheader
                st.subheader(f"Top {actual_causes_shown} Ranked Causes", anchor=False)

                # Each ranked cause in an expander
                for c_idx, cause in enumerate(causes_with_timestamps[:max_causes_to_show]):
                    if "bpm glossary" in cause.get("source_document","").lower():
                        continue
                        
                    # Get the datetime object again for display purposes
                    with st.container(border=True):
                        doc_name = cause.get('source_document', 'N/A')
                        dt_object = get_datetime_from_filename(cause.get("source_document", ""))
                        trigger_str = dt_object.strftime("%d.%m.%Y") if dt_object else "N/A"
                        
                        st.metric("Drift Trigger Date", trigger_str)
                        st.markdown(f"**Confidence:** {cause.get('confidence_score',0)*100:.1f}%")
                        
                        # Columns for document name and view button
                        col_doc_name, col_doc_button = st.columns([3, 1])

                        with col_doc_name:
                            st.markdown(f"**Source Document:** {doc_name}")
                        
                        with col_doc_button:
                            # Construct the correct path from project_root -> frontend -> static -> documents
                            doc_path = project_root / "frontend" / "static" / "documents" / doc_name
                            
                            if doc_name != 'N/A' and doc_path.exists():
                                with open(doc_path, "rb") as f:
                                    pdf_bytes = f.read()
                                
                                st.download_button(
                                    label="View Document",
                                    data=pdf_bytes,
                                    file_name=doc_name,
                                    type="secondary",
                                    key=f"doc_download_{idx}_{c_idx}"
                                )
                            else:
                                st.error("File not found.") # Add an error for clarity

                        exp_title = f"Cause #{c_idx+1}: {cause.get('context_category','N/A')}"
                        with st.expander(exp_title):
                            st.markdown(f"**Description:**")
                            st.markdown(f"{cause.get('cause_description','N/A')}")
                            st.markdown("**Evidence Snippet from the Source Document:**")
                            #st.code(cause.get("evidence_snippet",""), language="text")
                            #st.info(cause.get("evidence_snippet",""))
                            snippet = cause.get("evidence_snippet", "")
                            #st.markdown(f"> {snippet}")
                            #st.caption(f"Source Document: `{doc_name}`")
                            with st.container(border=True):
                                st.write(snippet)
                                
                            st.markdown(f"**Source Document:** `{doc_name}`")

                        # Feedback buttons
                        key_up   = f"up_{idx}_{c_idx}"
                        key_down = f"down_{idx}_{c_idx}"
                        fb_key   = (idx, c_idx)
                        if st.session_state.feedback_states.get(fb_key):
                            st.success("Thanks for your feedback!")
                        else:
                            col1, col2 = st.columns([1,6], gap="small")
                            with col1:
                                if st.button("👍", key=key_up):
                                    st.session_state.feedback_states[fb_key] = "positive"
                                    rec = {
                                        "timestamp": datetime.now().isoformat(),
                                        "feedback_type": "positive",
                                        "cause_rated": cause
                                    }
                                    fb_dir = project_root / "data" / "feedback"
                                    fb_dir.mkdir(exist_ok=True)
                                    with open(fb_dir/"feedback_log.jsonl","a") as f:
                                        f.write(json.dumps(rec)+"\n")
                                    st.rerun()
                            with col2:
                                if st.button("👎", key=key_down):
                                    st.session_state.feedback_states[fb_key] = "negative"
                                    rec = {
                                        "timestamp": datetime.now().isoformat(),
                                        "feedback_type": "negative",
                                        "cause_rated": cause
                                    }
                                    fb_dir = project_root / "data" / "feedback"
                                    fb_dir.mkdir(exist_ok=True)
                                    with open(fb_dir/"feedback_log.jsonl","a") as f:
                                        f.write(json.dumps(rec)+"\n")
                                    st.rerun()
                    st.write("") # Spacer between cause cards

else:
    st.info("Click ▶️ Run Full Analysis above to start.")

# This block adds the button to the sidebar and calls the dialog
if st.session_state.analysis_run_complete:
    with st.sidebar:
        #st.divider()
        st.subheader("Chatbot", anchor=False)
        if st.button("💬 Ask Follow-up Questions", use_container_width=True):
            st.session_state.show_chat = True # This flag opens the dialog

# This logic runs the dialog function (which is in your helper section)
# when the flag is set to True
if st.session_state.get("show_chat", False):
    run_chat_dialog()