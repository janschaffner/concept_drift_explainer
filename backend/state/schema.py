from typing import List, Dict, Any
from pydantic import BaseModel

class DriftMetadata(BaseModel):
    drift_id: str
    timestamp: str
    drift_type: str
    activity: str
    case_id: str

class SharedState(BaseModel):
    drift_metadata: DriftMetadata | None = None
    raw_context_snippets: List[Dict[str, Any]] = []
    filtered_snippets: List[Dict[str, Any]] = []
    classified_context: List[Dict[str, Any]] = []
    generated_explanation: str | None = None
    feedback: Dict[str, Any] = {}
