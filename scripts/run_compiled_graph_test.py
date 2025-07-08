import sys
import pprint
from pathlib import Path

# --- Path Correction ---
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
# -----------------------

from backend.graph.build_graph import build_graph

if __name__ == "__main__":
    print("--- Compiling and Running Full LangGraph Pipeline ---")

    # 1. Compile the graph from our builder function
    app = build_graph()

    # 2. UPDATED: Define the initial input for the graph
    # We must now specify which drift to analyze.
    # This example will run the first drift (index 0) from the first row (index 0).
    initial_input = {
        "selected_drift": {"row_index": 0, "drift_index": 0}
    }

    # 3. Invoke the graph and run the full pipeline
    print(f"\nInvoking the graph for drift: {initial_input['selected_drift']}...")
    final_state = app.invoke(initial_input)

    # 4. Print the final state
    print("\n--- Final State from LangGraph Execution ---")
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(final_state)

    print("\n--- End of Test ---")