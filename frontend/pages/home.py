import sys
from pathlib import Path
import warnings
import pandas as pd
import json
import ast
from datetime import datetime

import streamlit as st

# ── Path Correction ───────────────────────────────────────────────────────────────
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
# ──────────────────────────────────────────────────────────────────────────────────

# Suppress LangChain Pydantic V1 deprecation warning
warnings.filterwarnings("ignore", message=".*Pydantic BaseModel V1.*")

# ── Backend Imports ───────────────────────────────────────────────────────────────
from backend.graph.build_graph import build_graph
from backend.agents.chatbot_agent import run_chatbot_agent
from backend.agents.drift_linker_agent import run_drift_linker_agent, ConnectionType
# ──────────────────────────────────────────────────────────────────────────────────

# ── Descriptions for Drift Linker Categories ─────────────────────────────────────
CONNECTION_TYPE_DESCRIPTIONS = {
    ConnectionType.STRONG_CAUSAL.value:   "One drift appears to be a direct cause of another subsequent drift.",
    ConnectionType.SHARED_EVIDENCE.value: "The drifts are linked by common evidence or appear to share the same underlying root cause.",
    ConnectionType.THEMATIC_OVERLAP.value:"The drifts are not directly linked but share a similar theme or occur in related business areas.",
}
# ──────────────────────────────────────────────────────────────────────────────────

# ── Helper: Parse date out of filename ────────────────────────────────────────────
def get_date_from_filename(filename: str) -> str:
    try:
        name = Path(filename).name
        date_str = name.split('_')[0]
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.strftime("%d.%m.%Y")
    except Exception:
        return "N/A"
# ──────────────────────────────────────────────────────────────────────────────────

# ── Helper: Load & unpack drifts from drift_outputs/*.csv ────────────────────────
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

# ──────────────────────────────────────────────────────────────────────────────────

# ── Session State Initialization ────────────────────────────────────────────────
def init_session_state():
    st.session_state.all_explanations    = []
    st.session_state.error_message       = None
    st.session_state.feedback_states     = {}
    st.session_state.chat_history        = []
    st.session_state.full_state_log      = []
    st.session_state.linked_drift_summary= None
    st.session_state.connection_type     = None
    st.session_state.analysis_run_complete = False
    st.session_state.show_chat           = False

if "analysis_run_complete" not in st.session_state:
    init_session_state()
# ──────────────────────────────────────────────────────────────────────────────────

# ── Page Config & Title ─────────────────────────────────────────────────────────
st.set_page_config(page_title="Concept Drift Explainer", page_icon="🤖", layout="wide")
st.title("Concept Drift Explainer")
# ──────────────────────────────────────────────────────────────────────────────────

# ── Hallucination Warning ───────────────────────────────────────────────────────
st.warning(
    "The Concept Drift Explainer can make mistakes. Verify important information!",
    icon="⚠️"
)
# ──────────────────────────────────────────────────────────────────────────────────

# ── Top Controls: Event‐Log Dropdown + Run Button ────────────────────────────────
logs_folder = project_root / "data" / "event_logs"
log_files = sorted([p.name for p in logs_folder.iterdir() if p.is_dir()])

col_left, col_right = st.columns(2, gap="large")  # equal widths
with col_left:
    drift_options = load_and_unpack_drifts(st.session_state.selected_log)
    n_drifts = len(drift_options)
    selected_log = st.selectbox(
        "Select Event Log to Analyze",
        options=log_files,
        help="Choose one event log file",
        key="selected_log"
    )
    st.markdown(
        f"""
        <div style="
          background-color: #d4edda;
          color: #155724;
          padding: 0.5rem 1rem;
          border-radius: 0.5rem;
          margin-bottom: 0.5rem;
        ">
          <strong>{n_drifts} drift(s) found</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_right:
    if st.button("▶️ Run Drift Analysis", disabled=(n_drifts == 0)):
        # Reset state
        init_session_state()
        all_expls = []
        full_log  = []
        progress   = st.progress(0.0, text="Starting analysis…")
        # Run backend graph pipeline
        with st.spinner(f"Analyzing {n_drifts} drift(s)... This may take a few minutes."):
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
# ──────────────────────────────────────────────────────────────────────────────────

# ── Once analysis is done: divider + tabs for each drift ─────────────────────────
st.divider()

if st.session_state.error_message:
    st.error(f"An error occurred: {st.session_state.error_message}")

elif st.session_state.all_explanations:
    exps = st.session_state.all_explanations
    st.success(f"Successfully analyzed {len(exps)} drift(s).")

    # Ask‐follow‐up button
    if st.button("💬 Ask Follow-up Questions"):
        st.session_state.show_chat = True

    # Floating chat dialog (modal)
    @st.dialog("Conversational Analysis")
    def run_chat_dialog():
        if not st.session_state.chat_history:
            st.session_state.chat_history.append(
                ("assistant", "Hello! Ask me anything about the analysis.")
            )
        for author, msg in st.session_state.chat_history:
            avatar = "frontend/assets/user_avatar.png" if author=="user" else "frontend/assets/chatbot_avatar.png"
            with st.chat_message(author, avatar=avatar):
                st.markdown(msg)

        if prompt := st.chat_input("Your question..."):
            st.session_state.chat_history.append(("user", prompt))
            with st.chat_message("assistant", avatar="frontend/assets/chatbot_avatar.png"):
                with st.spinner("Thinking..."):
                    resp = run_chatbot_agent({
                        "full_state_log": st.session_state.full_state_log,
                        "chat_history":   st.session_state.chat_history,
                        "user_question":  prompt
                    })
                    st.session_state.chat_history = resp["chat_history"]
                    st.markdown(st.session_state.chat_history[-1][1])
            st.rerun()

    if st.session_state.show_chat:
        run_chat_dialog()

    # Cross‐drift summary
    if st.session_state.linked_drift_summary:
        ctype = st.session_state.connection_type
        desc  = CONNECTION_TYPE_DESCRIPTIONS.get(ctype, "")
        with st.container():
            st.subheader("🔗 Cross-Drift Analysis")
            st.markdown(f"**Type:** {ctype}")
            st.caption(desc)
            st.markdown(st.session_state.linked_drift_summary)
            st.markdown("---")

    # Tabs: one per drift explanation (default to first)
    labels = [f"Drift #{i+1}" for i in range(len(exps))]
    tabs   = st.tabs(labels)

    for idx, tab in enumerate(tabs):
        with tab:
            explanation = exps[idx]
            state       = st.session_state.full_state_log[idx]
            info        = state.get("drift_info", {})

            # Header with type & timeframe
            dtype = info.get("drift_type", "Unknown").capitalize()
            start = info.get("start_timestamp","N/A").split(" ")[0]
            end   = info.get("end_timestamp","N/A").split(" ")[0]
            try:
                s_fmt = datetime.strptime(start, "%Y-%m-%d").strftime("%d.%m.%Y")
                e_fmt = datetime.strptime(end,   "%Y-%m-%d").strftime("%d.%m.%Y")
            except:
                s_fmt, e_fmt = "N/A","N/A"

            st.subheader(f"Explanation for {dtype}")
            st.markdown(f"<h4 style='margin-top:-0.75rem;'>Timeframe: {s_fmt} – {e_fmt}</h4>",
                        unsafe_allow_html=True)
            st.info(explanation.get("summary", "No summary available."))

            # Each ranked cause in an expander
            for c_idx, cause in enumerate(explanation.get("ranked_causes", [])):
                if "bpm glossary" in cause.get("source_document","").lower():
                    continue
                with st.container():
                    trigger = get_date_from_filename(cause.get("source_document",""))
                    st.metric("Drift Trigger Date", trigger)
                    st.markdown(f"**Confidence:** {cause.get('confidence_score',0)*100:.1f}%")
                    st.markdown(f"**Source Document:** {cause.get('source_document','N/A')}")

                    exp_title = f"Cause #{c_idx+1}: {cause.get('context_category','N/A')}"
                    with st.expander(exp_title):
                        st.markdown(f"**Description:** {cause.get('cause_description','N/A')}")
                        st.markdown("**Evidence:**")
                        st.code(cause.get("evidence_snippet",""), language="text")

                    # Feedback buttons
                    key_up   = f"up_{idx}_{c_idx}"
                    key_down = f"down_{idx}_{c_idx}"
                    fb_key   = (idx, c_idx)
                    if st.session_state.feedback_states.get(fb_key):
                        st.success("Thanks for your feedback!")
                    else:
                        col1, col2 = st.columns([1,1], gap="small")
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

            st.divider()

else:
    st.info("Click ▶️ Run Full Analysis above to start.")
