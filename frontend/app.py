import streamlit as st
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# --- Construct the absolute path to the icon ---
icon_path = project_root / "frontend" / "assets" / "tab_icon.png"

# --- Set Page Config (global) ---
st.set_page_config(
    page_title="Concept Drift Explainer",
    page_icon=str(icon_path),  # Use the full, correct path
    layout="wide"
)

def init_session_state():
    """Initializes session state variables if they don't exist."""
    if "state_initialized" not in st.session_state:
        st.session_state.state_initialized = True
        
        # --- Settings ---
        st.session_state.max_causes = 3 # Default value
        st.session_state.confidence_threshold = 0.25
        
        # --- Analysis Results & UI State ---
        st.session_state.all_explanations = []
        st.session_state.error_message = None
        st.session_state.feedback_states = {}
        st.session_state.chat_history = []
        st.session_state.full_state_log = []
        st.session_state.linked_drift_summary = None
        st.session_state.connection_type = None
        st.session_state.analysis_run_complete = False
        st.session_state.show_chat = False

# Call the initialization function at the top of the script
init_session_state()

# Define the pages
pages = [
    st.Page("pages/home.py",     title="Home",             icon="🏠"),
    st.Page("pages/manage_context.py",   title="Manage Context",   icon="📁"),
    st.Page("pages/settings.py", title="Settings",         icon="⚙️"),
]

# Get a reference to the home page object from our list
home_page = pages[0]

# Run the navigation
page = st.navigation(pages, position="sidebar")

# Reset the chat flag when navigating away from the Home page.
# The 'page' object tells us which page script is currently active.
# We check if the active page is NOT the home page.
if page is not home_page:
    # If we are on any other page, ensure the chat dialog flag is reset.
    st.session_state.show_chat = False

# Run the selected page
page.run()