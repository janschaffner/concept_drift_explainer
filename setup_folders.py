import os

def write_file(path, content=""):
    with open(path, "w") as f:
        f.write(content)

# Folder structure (relative to current directory)
folders = [
    "backend/agents",
    "backend/graph",
    "backend/state",
    "backend/utils",
    "frontend",
    "data/event_logs",
    "data/drift_outputs",
    "data/documents",
    "tests"
]

# Create folders and __init__.py files
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    write_file(os.path.join(folder, "__init__.py"), f"# Init for {folder.split('/')[-1]}")

# Backend agent files
agent_files = {
    "backend/agents/drift_agent.py": "# TODO: implement drift_agent\n",
    "backend/agents/context_retrieval_agent.py": "# TODO: implement context_retrieval_agent\n",
    "backend/agents/context_mapper_agent.py": "# TODO: implement context_mapper_agent\n",
    "backend/agents/franzoi_mapper_agent.py": "# TODO: implement franzoi_mapper_agent\n",
    "backend/agents/explanation_agent.py": "# TODO: implement explanation_agent\n",
    "backend/agents/evaluation_agent.py": "# TODO: implement evaluation_agent\n",
    "backend/agents/chatbot_agent.py": "# TODO: implement chatbot_agent\n"
}

# Backend utility files
utility_files = {
    "backend/utils/embedding.py": "# TODO: document embedding logic\n",
    "backend/utils/retrieval.py": "# TODO: semantic search logic\n",
    "backend/utils/timestamp_utils.py": "# TODO: date comparison and filtering helpers\n",
    "backend/utils/logging_config.py": "# TODO: structured logging setup\n"
}

# Backend graph and state
write_file("backend/run_pipeline.py", """\
# Entrypoint script for running the LangGraph pipeline
from backend.graph.build_graph import build_drift_explainer_graph

if __name__ == "__main__":
    graph = build_drift_explainer_graph()
    graph.invoke({})
""")

write_file("backend/graph/build_graph.py", """\
# Define LangGraph pipeline here
def build_drift_explainer_graph():
    # TODO: import agents and define graph structure
    pass
""")

write_file("backend/state/schema.py", """\
from typing import List, Dict, Any
from pydantic import BaseModel

class DriftMetadata(BaseModel):
    drift_id: str
    timestamp: str
    drift_type: str
    activity: str
    case_id: str

class SharedState(BaseModel):
    drift_metadata: DriftMetadata | None = None
    raw_context_snippets: List[Dict[str, Any]] = []
    filtered_snippets: List[Dict[str, Any]] = []
    classified_context: List[Dict[str, Any]] = []
    generated_explanation: str | None = None
    feedback: Dict[str, Any] = {}
""")

# Frontend files
write_file("frontend/streamlit_ui.py", """\
import streamlit as st

st.title("Context-Aware Drift Explanation Tool")

# TODO: display explanation and chat interface
""")

write_file("frontend/cli.py", """\
def run_cli():
    print("Running CLI interface...")
""")

# Test files
write_file("tests/test_drift_agent.py", "def test_drift_agent(): assert True\n")
write_file("tests/test_explanation_agent.py", "def test_explanation_agent(): assert True\n")

# Write all agent and utility files
for path, content in {**agent_files, **utility_files}.items():
    write_file(path, content)

print("✅ Full project structure created with all .py files (no outer folder).")