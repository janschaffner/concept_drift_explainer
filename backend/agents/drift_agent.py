import os
import sys
import logging
from pathlib import Path
import pandas as pd
import ast
import json
from lxml import etree
import re
from nltk.stem import PorterStemmer

# --- Path Correction ---
# Ensures that the script can correctly import modules from the 'backend' directory
# by adding the project's root directory to the system's path.
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
# -----------------------

# Import the graph state schema.
from backend.state.schema import GraphState, DriftInfo

# Configure basic logging.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def build_activity_to_timestamp_map(window_info: dict) -> dict:
    """
    Pre-processes the window_info.json data into a direct lookup table for efficiency.
    This allows for quick retrieval of timestamps based on activity instance IDs.
    """
    activity_map = {}
    # Assumes the JSON has a single root key which is the process name.
    process_name = next(iter(window_info))
    for item in window_info[process_name].values():
        activity_name, timestamp = item
        activity_map[activity_name] = timestamp
    return activity_map

# Keyword Extraction to enrich VecDB Query
def extract_general_keywords(trace: etree._Element) -> list:
    """Extracts a flat list of general keywords from an XES trace."""
    general_keywords = set()
    st = PorterStemmer() # Used to normalize keywords (e.g., "running" -> "run")
    
    # Extract general keywords from trace attributes
    for attr in trace.xpath(".//string | .//int | .//float"):
        val = attr.get("value", "")
        if val and val not in {"UNKNOWN", "MISSING", "EMPTY"} and not any(char.isdigit() for char in val) and len(val) > 3:
            for word in re.split(r'[\s,()/-]', val):
                if word and len(word) > 3:
                    general_keywords.add(st.stem(word.lower()))

    for event in trace.findall('event'):
        for key in ["concept:name", "activityNameEN"]:
            name_element = event.find(f"string[@key='{key}']")
            if name_element is not None:
                label = name_element.get("value", "").strip().lower()
                if label:
                    for word in re.split(r'[\s_]', label):
                        if word and word.isalpha() and len(word) > 3:
                            general_keywords.add(st.stem(word))

    return list(general_keywords)

def run_drift_agent(state: GraphState) -> dict:
    """
    Parses a selected drift, extracts key information, and creates a semantic
    drift phrase to populate the initial state.
    """
    logging.info("--- Running Drift Agent ---")
    selection = state.get("selected_drift")
    if not selection:
        return {"error": "No drift was selected."}
    
    data_dir = selection.get("data_dir")
    data_dir = Path(data_dir) if data_dir and Path(data_dir).exists() else project_root / "data" / "drift_outputs"
    logging.info(f"Using data directory: {data_dir}")

    try:
        csv_path = next(data_dir.glob("*.csv"))
        json_path = next(data_dir.glob("*.json"))
        xes_path = next(data_dir.glob("*.xes"))
        drift_df = pd.read_csv(csv_path)
        with open(json_path, 'r') as f: window_info = json.load(f)
        log_tree = etree.parse(xes_path)
        traces = log_tree.xpath("//trace")
    except (StopIteration, Exception) as e:
        return {"error": f"Failed to load or find data files in {data_dir}: {e}"}

    row_index = selection.get("row_index", 0)
    drift_index_in_row = selection.get("drift_index", 0)
    selected_row = drift_df.iloc[row_index]
    trace_to_analyze = traces[row_index]
    
    logging.info(f"ℹ️  Processing drift #{drift_index_in_row + 1} from CSV row #{row_index + 1}  ℹ️")
    
    try:
        all_changepoints = ast.literal_eval(selected_row["Detected Changepoints"])
        changepoint_pair = all_changepoints[drift_index_in_row]
        
        general_keywords = extract_general_keywords(trace_to_analyze)

        # Preserve order when creating the drift phrase
        ordered_concepts = []
        seen_concepts = set()
        for activity in changepoint_pair:
            words = re.findall(r"[A-Za-z]+", activity.lower())
            if words:
                concept = " ".join(words)
                if concept not in seen_concepts:
                    ordered_concepts.append(concept)
                    seen_concepts.add(concept)
        drift_phrase = " ".join(ordered_concepts)

        all_drift_types = ast.literal_eval(selected_row["Detected Drift Types"])
        drift_type = all_drift_types[drift_index_in_row]
        confidence_str = selected_row["Prediction Confidence"].strip('[]')
        all_confidences = [float(c) for c in confidence_str.split()]
        confidence = all_confidences[drift_index_in_row]
    except (ValueError, SyntaxError, IndexError) as e:
        return {"error": f"Could not parse or index drift data from CSV: {e}"}
    
    activity_map = build_activity_to_timestamp_map(window_info)
    start_timestamp = activity_map.get(changepoint_pair[0])
    end_timestamp = activity_map.get(changepoint_pair[1])

    if not all([start_timestamp, end_timestamp]):
        return {"error": f"Could not find timestamps for changepoint pair: {changepoint_pair}"}
    
    drift_info: DriftInfo = {
        "process_name": xes_path.stem, 
        "changepoints": changepoint_pair,
        "drift_type": drift_type, 
        "confidence": confidence,
        "start_timestamp": start_timestamp, 
        "end_timestamp": end_timestamp,
    }
    
    if "gold_doc" in selection:
        drift_info["gold_doc"] = selection["gold_doc"]

    logging.info(f"Populated drift_info: {drift_info}")
    logging.info(f"Extracted {len(general_keywords)} general keywords (sample): {general_keywords[:10]}...")
    logging.info(f"Generated semantic drift phrase = '{drift_phrase}'")
    
    return {
        "drift_info": drift_info,
        "drift_keywords": general_keywords,
        "drift_phrase": drift_phrase,
    }