import streamlit as st

# 1) Page config
st.set_page_config(page_title="Settings", layout="wide")
st.title("⚙️ Settings")
st.divider()

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