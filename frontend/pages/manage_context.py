import streamlit as st
from pathlib import Path
import os
from datetime import datetime
import pandas as pd

# --- Import the backend functions ---
from backend.utils.ingest_documents import process_context_files, initialize_ingestion_backend

# --- Path Setup ---
# This correctly finds the project's 'frontend' directory,
# which is the root for the running Streamlit app.
project_root = Path(__file__).resolve().parents[2]
DOCUMENTS_DIR = project_root / "frontend" / "static" / "documents"
# Ensure the directory exists
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

# --- Initialize Backend and Cache Resources ---
# @st.cache_resource ensures this expensive setup runs only once.
@st.cache_resource
def get_ingestion_resources():
    return initialize_ingestion_backend()

index, embedder, text_splitter = get_ingestion_resources()

# --- Page Config & Title ---
#st.set_page_config(page_title="Upload & Manage Context", layout="wide")
st.title("📁 Upload & Manage Context Documents")
st.write("Upload new documents or review the current context knowledge base.")

# Display the success message from the previous run
if "upload_status" in st.session_state:
    st.success(st.session_state.upload_status)
    # Clear the message so it doesn't show up on subsequent reloads
    del st.session_state.upload_status

st.divider()

# --- 1. File Uploader Section (Wrapped in a Form) ---
with st.container(border=True):
    st.subheader("Upload New Documents")
    
    with st.form(key="upload_form", clear_on_submit=True):
        uploaded_files = st.file_uploader(
            "Select one or more documents to add to the knowledge base.",
            accept_multiple_files=True,
            type=['pdf', 'docx', 'txt', 'pptx', 'png', 'jpg']
        )
        
        # The submit button is no longer disabled
        submitted = st.form_submit_button(
            "Process Uploaded Files",
            use_container_width=True
        )
        
        # Check for submitted files AFTER the button is clicked
        if submitted and uploaded_files:
            saved_file_paths = []
            with st.spinner("Saving and processing new documents..."):
                for uploaded_file in uploaded_files:
                    file_path = DOCUMENTS_DIR / uploaded_file.name
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    saved_file_paths.append(file_path)
                
                process_context_files(saved_file_paths, index, embedder, text_splitter)
            
            st.session_state.upload_status = f"✅ Successfully processed {len(saved_file_paths)} new document(s)!"
            st.cache_data.clear()
            st.rerun()
        elif submitted:
            # Optional: Show a warning if the user clicks with no files
            st.warning("Please select at least one file to upload.")

st.write("") # Spacer

# --- 2. Current Document Inventory (Enhanced Version) ---
with st.container(border=True):
    st.subheader("Current Knowledge Base")
    try:
        doc_paths = [DOCUMENTS_DIR / f for f in os.listdir(DOCUMENTS_DIR)]
        
        if not doc_paths:
            st.info("No documents have been uploaded yet.")
        else:
            doc_info = []
            for path in doc_paths:
                stat = path.stat()
                doc_info.append({
                    "Filename": path.name,
                    "Size (KB)": f"{stat.st_size / 1024:.2f}",
                    "Last Modified": datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                })
            
            # Create a pandas DataFrame
            df = pd.DataFrame(doc_info)
            
            # Calculate the required height
            # (Number of rows + 1 for header) * 35 pixels per row
            table_height = (len(df) + 1) * 35 + 3
            
            # Display the interactive table with the calculated height
            st.dataframe(
                df, 
                use_container_width=True,
                height=table_height
            )

    except FileNotFoundError:
        st.error(f"Error: The directory was not found at {DOCUMENTS_DIR}")