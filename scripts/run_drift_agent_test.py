import sys
import pprint
from pathlib import Path

# --- Path Correction ---
# Add the project's root directory (the parent of 'scripts') to the Python path
# This allows us to import from the 'backend' package as if we were running from the root
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
# -----------------------

from backend.agents.drift_agent import run_drift_agent

if __name__ == "__main__":
    print("--- Starting Agent Test (from scripts folder) ---")

    # The graph always starts with an initial state.
    # For the Drift Agent, the initial state can be empty.
    initial_state = {}

    # Call the agent function directly
    result = run_drift_agent(initial_state)

    print("\n--- Agent Execution Result ---")
    # Use pprint for a nicely formatted output of the dictionary
    pprint.pprint(result)
    print("--- End of Agent Test ---")