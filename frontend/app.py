import streamlit as st

def init_session_state():
    """Initializes session state variables if they don't exist."""
    if "state_initialized" not in st.session_state:
        st.session_state.state_initialized = True
        
        # --- Settings ---
        st.session_state.max_causes = 5 # Default value
        
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
    st.Page("pages/home.py",     title="Home",     icon="🏠"),
    st.Page("pages/upload.py",   title="Upload",   icon="📁"),
    st.Page("pages/settings.py", title="Settings", icon="⚙️"),
]

# Run the navigation
page = st.navigation(pages, position="sidebar")
page.run()