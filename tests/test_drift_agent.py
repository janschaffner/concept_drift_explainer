import sys
import pprint
from pathlib import Path

# --- Path Correction ---
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
# -----------------------

# CORRECTED: Import the agent function directly
from backend.agents.drift_agent import run_drift_agent
from backend.state.schema import GraphState

def run_agent_test():
    """
    Directly invokes the run_drift_agent function to test its output.
    """
    print("--- Testing Drift Agent Output ---")

    # 1. Define the input for the drift to be tested
    test_set_path = project_root / "data" / "event_logs" / "BPI2020_InternationalDeclarations"
    
    initial_state = GraphState(
        selected_drift={
            "row_index": 0,
            "drift_index": 0,
            "data_dir": str(test_set_path)
        }
    )

    # 2. --- CORRECTED: Invoke the agent's function directly ---
    print(f"\nInvoking only the Drift Agent for test set: {test_set_path.name}...")
    output_dict = run_drift_agent(initial_state)

    # 3. Print the output to verify the new keys
    print("\n--- Output from Drift Agent ---")
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(output_dict)

    # 4. Verify the new keys exist in the agent's output
    assert "error" not in output_dict, f"Agent returned an error: {output_dict.get('error')}"
    assert "drift_info" in output_dict, "drift_info key is missing!"
    assert "drift_keywords" in output_dict, "drift_keywords key is missing!"
    
    print("\n✅ Test Passed: All new keys are present in the agent's output.")


if __name__ == "__main__":
    run_agent_test()