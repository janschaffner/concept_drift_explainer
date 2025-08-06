import streamlit as st
from backend.utils.cache import clear_llm_cache

# Display the toast message for cleared cache from the previous run
if "cache_status_message" in st.session_state:
    st.toast(st.session_state.cache_status_message)
    # Clear the message so it only appears once
    del st.session_state.cache_status_message

st.title("⚙️ Settings")
st.divider()

# --- Main Settings ---

st.subheader("Main Settings")

# This callback function runs IMMEDIATELY when the selectbox value changes.
def update_setting():
    # It takes the value from the widget (st.session_state.max_causes_widget)
    # and saves it to our persistent state variable (st.session_state.max_causes).
    st.session_state.max_causes = st.session_state.max_causes_widget

# Create two columns; the first will be narrow, the second will be empty space.
col1, col2 = st.columns([1, 2])
with col1:
    options = [3, 4, 5]
    current_index = options.index(st.session_state.max_causes)

    st.selectbox(
        label="Maximum causes to display per drift",
        options=options,
        index=current_index,
        key="max_causes_widget", # A unique key for the widget itself
        on_change=update_setting  # Run our function on change
    )

# --- Cache Management ---

st.divider()
st.subheader("Cache Management")

# This function defines the content of our confirmation dialog
@st.dialog("Confirm Deletion")
def confirm_clear_cache():
    st.warning("Are you sure you want to permanently delete the LLM cache?")
    st.caption("This action cannot be undone.")
    
    # Create columns for the "Yes" and "Cancel" buttons
    col1, col2 = st.columns(2)
    if col1.button("Yes, clear the cache", type="primary"):
        status_message = clear_llm_cache()
        # Save message to state instead of showing it here
        st.session_state.cache_status_message = status_message
        st.session_state.show_confirm_dialog = False
        st.rerun()

    if col2.button("Cancel"):
        st.session_state.show_confirm_dialog = False
        st.rerun()

# This is the main button the user first clicks
if st.button(
    "Clear LLM Cache",
    type="primary",
    help="This will open a confirmation prompt before deleting the cache."
):
    # Set a flag in session state to show our dialog
    st.session_state.show_confirm_dialog = True

# This line checks the flag and runs the dialog function if it's true
if st.session_state.get("show_confirm_dialog", False):
    confirm_clear_cache()