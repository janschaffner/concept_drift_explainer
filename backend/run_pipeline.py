# Entrypoint script for running the LangGraph pipeline
from backend.graph.build_graph import build_drift_explainer_graph

if __name__ == "__main__":
    graph = build_drift_explainer_graph()
    graph.invoke({})
