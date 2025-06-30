import sys
import pprint
from pathlib import Path

# --- Path Correction ---
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
# -----------------------

from backend.agents.drift_agent import run_drift_agent
from backend.agents.context_retrieval_agent import run_context_retrieval_agent
from backend.agents.franzoi_mapper_agent import run_franzoi_mapper_agent
from backend.agents.explanation_agent import run_explanation_agent # <-- NEW IMPORT

if __name__ == "__main__":
    print("--- Starting Full Agent Chain Test ---")

    # 1. Start with an empty initial state
    state = {}
    pp = pprint.PrettyPrinter(indent=2)

    # 2. Run the Drift Agent
    print("\n[Step 1] Running Drift Agent...")
    state.update(run_drift_agent(state))
    if state.get("error"):
        print(f"ERROR: {state['error']}"); sys.exit(1)
    print("Drift Agent completed successfully.")

    # 3. Run the Context Retrieval Agent
    print("\n[Step 2] Running Context Retrieval Agent...")
    state.update(run_context_retrieval_agent(state))
    if state.get("error"):
        print(f"ERROR: {state['error']}"); sys.exit(1)
    print(f"Context Retrieval Agent completed successfully. Found {len(state.get('raw_context_snippets', []))} snippets.")

    # 4. Run the Franzoi Mapper Agent
    print("\n[Step 3] Running Franzoi Mapper Agent...")
    state.update(run_franzoi_mapper_agent(state))
    if state.get("error"):
        print(f"ERROR: {state['error']}"); sys.exit(1)
    print("Franzoi Mapper Agent completed successfully.")

    # 5. Run the Explanation Agent <-- NEW STEP
    print("\n[Step 4] Running Explanation Agent...")
    state.update(run_explanation_agent(state))
    if state.get("error"):
        print(f"ERROR: {state['error']}"); sys.exit(1)
    print("Explanation Agent completed successfully.")


    # 6. Print the final, fully enriched state
    print("\n--- Final State ---")
    pp.pprint(state)

    print("\n--- End of Full Agent Chain Test ---")