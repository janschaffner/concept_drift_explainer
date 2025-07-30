import streamlit as st

pages = [
    st.Page("pages/home.py",     title="Home",     icon="🏠"),
    st.Page("pages/upload.py",   title="Upload",   icon="📁"),
    st.Page("pages/settings.py", title="Settings", icon="⚙️"),
]

page = st.navigation(pages, position="sidebar", expanded=True)
page.run()
