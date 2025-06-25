import sys
import pprint
from pathlib import Path

# --- Path Correction ---
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
# -----------------------

from backend.agents.drift_agent import run_drift_agent
from backend.agents.context_retrieval_agent import run_context_retrieval_agent

if __name__ == "__main__":
    print("--- Starting Agent Chain Test ---")

    # 1. Start with an empty initial state
    state = {}

    # 2. Run the Drift Agent
    print("\n[Step 1] Running Drift Agent...")
    drift_result = run_drift_agent(state)
    state.update(drift_result)

    # Check for errors before proceeding
    if state.get("error"):
        print(f"Error after Drift Agent: {state['error']}")
    else:
        print("Drift Agent completed successfully.")
        pprint.pprint(state['drift_info'])

        # 3. Run the Context Retrieval Agent with the updated state
        print("\n[Step 2] Running Context Retrieval Agent...")
        retrieval_result = run_context_retrieval_agent(state)
        state.update(retrieval_result)

        print("\n--- Final State ---")
        pprint.pprint(state)

    print("\n--- End of Agent Chain Test ---")