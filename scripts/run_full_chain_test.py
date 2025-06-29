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

if __name__ == "__main__":
    print("--- Starting Full Agent Chain Test ---")

    # 1. Start with an empty initial state
    state = {}
    pp = pprint.PrettyPrinter(indent=2)

    # 2. Run the Drift Agent
    print("\n[Step 1] Running Drift Agent...")
    drift_result = run_drift_agent(state)
    state.update(drift_result)

    if state.get("error"):
        print(f"Error after Drift Agent: {state['error']}")
        sys.exit(1) # Exit if there's an error
    print("Drift Agent completed successfully.")

    # 3. Run the Context Retrieval Agent
    print("\n[Step 2] Running Context Retrieval Agent...")
    retrieval_result = run_context_retrieval_agent(state)
    state.update(retrieval_result)
    
    if state.get("error"):
        print(f"Error after Context Retrieval Agent: {state['error']}")
        sys.exit(1)
    print(f"Context Retrieval Agent completed successfully. Found {len(state.get('raw_context_snippets', []))} snippets.")


    # 4. Run the Franzoi Mapper Agent
    print("\n[Step 3] Running Franzoi Mapper Agent...")
    classification_result = run_franzoi_mapper_agent(state)
    state.update(classification_result)

    if state.get("error"):
        print(f"Error after Franzoi Mapper Agent: {state['error']}")
        sys.exit(1)
    print("Franzoi Mapper Agent completed successfully.")


    # 5. Print the final, fully enriched state
    print("\n--- Final State ---")
    pp.pprint(state)

    print("\n--- End of Full Agent Chain Test ---")