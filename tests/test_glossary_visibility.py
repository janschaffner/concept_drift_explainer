import sys
from pathlib import Path
import pytest

# --- Path Correction ---
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
# -----------------------

from backend.state.schema import GraphState
from backend.agents.explanation_agent import run_explanation_agent

def test_glossary_not_in_evidence():
    """
    Tests that snippets from the 'BPM Glossary' are used for reasoning
    but are NOT cited as evidence in the final explanation.
    """
    # 1. Arrange: Fabricate one glossary and one context snippet
    glossary_snip = {
        "snippet_text": "Sudden Drift – Abrupt replacement of one process variant with another.",
        "source_document": "BPM Glossary",
        "classifications": [{"full_path": "BPM-KB", "reasoning": ""}]
    }
    context_snip = {
        "snippet_text": "Memo: Effective immediately, all payment requests will use the new 'FastTrack' system.",
        "source_document": "2024-01-15_Emergency_Directive.pdf",
        "classifications": [{"full_path": "ORGANIZATION_INTERNAL::IT_Management", "reasoning": ""}]
    }

    # Create a state object that simulates the output of the Re-Ranker Agent
    test_state = GraphState(
        reranked_context_snippets=[context_snip],
        supporting_context=[glossary_snip],
        drift_info={
            "drift_type": "sudden",
            "start_timestamp": "2024-01-15T12:00:00",
            "end_timestamp": "2024-01-15T12:00:00"
        }
    )

    # 2. Act
    result = run_explanation_agent(test_state)

    # 3. Assert
    assert "error" not in result, f"Agent returned an error: {result.get('error')}"
    assert "explanation" in result and "ranked_causes" in result["explanation"]
    
    # The critical test: ensure no cause cites the glossary as its source
    for cause in result["explanation"]["ranked_causes"]:
        assert cause["source_document"] != "BPM Glossary", \
            "A cause in the final explanation incorrectly cited the BPM Glossary as evidence."

    print("\nSUCCESS: Test confirmed that glossary items are not cited as evidence.")