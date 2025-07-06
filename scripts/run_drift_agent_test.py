# scripts/test_specific_drift.py

import sys
import pprint
from pathlib import Path

# --- Path Correction ---
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
# -----------------------

from backend.graph.build_graph import build_graph

if __name__ == "__main__":
    print("--- Testing analysis of a specific drift ---")

    app = build_graph()

    # Manually specify which drift to analyze.
    # This example tests the second drift (index 1) within the first row (index 0) of the CSV.
    # Change these values to test other drifts.
    drift_selection_to_test = {"row_index": 0, "drift_index": 1}
    
    # The initial input for the graph must now match the expected state key.
    initial_input = {"selected_drift": drift_selection_to_test}

    print(f"\nInvoking graph for drift: {drift_selection_to_test}...")
    final_state = app.invoke(initial_input)

    print("\n--- Final State from LangGraph Execution ---")
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(final_state)

    print("\n--- End of Test ---")