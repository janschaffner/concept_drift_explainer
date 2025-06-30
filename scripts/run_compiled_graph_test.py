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

    # 2. Define the initial input for the graph
    # Our graph starts empty and the first agent loads the data from files.
    initial_input = {}

    # 3. Invoke the graph and run the full pipeline
    print("\nInvoking the graph...")
    final_state = app.invoke(initial_input)

    # 4. Print the final state
    print("\n--- Final State from LangGraph Execution ---")
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(final_state)

    print("\n--- End of Test ---")