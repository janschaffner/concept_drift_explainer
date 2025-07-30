import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

import streamlit as st
from backend.utils.ingest_documents import process_context_files

# Page configuration
st.set_page_config(
    page_title="Upload Context Documents",
    layout="wide",
)

st.title("📄 Upload Context Documents")

st.divider()
#st.header("Add New Context")
uploaded_files = st.file_uploader(
    "Upload new documents (.pdf, .pptx, .docx, .txt, .png, .jpg)",
    accept_multiple_files=True,
    type=["pdf", "pptx", "docx", "txt", "png", "jpg"],
    key="uploaded_context_files"
)

if st.button("Process Uploaded Files") and uploaded_files:
    saved_files_paths = []
    with st.spinner("Saving and processing uploaded files..."):
        # Ensure documents folder exists
        DOCUMENTS_PATH = project_root / "data" / "documents"
        DOCUMENTS_PATH.mkdir(parents=True, exist_ok=True)
        for uploaded_file in uploaded_files:
            file_path = DOCUMENTS_PATH / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            saved_files_paths.append(file_path)

        # Call backend ingestion logic
        process_context_files(saved_files_paths)

    st.success(f"Successfully processed {len(saved_files_paths)} document(s)!")
    # Clear cache and rerun UI
    st.cache_data.clear()
    st.rerun()
