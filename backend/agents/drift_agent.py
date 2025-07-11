import os
import sys
import logging
from pathlib import Path
import pandas as pd
import ast
import json
import pm4py
from lxml import etree
import re
from nltk.stem import PorterStemmer

# --- Path Correction ---
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
# -----------------------

# Import our graph state schema
from backend.state.schema import GraphState, DriftInfo

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def build_activity_to_timestamp_map(window_info: dict) -> dict:
    """
    Pre-processes the window_info.json data into a direct lookup table for efficiency.
    """
    activity_map = {}
    process_name = next(iter(window_info))
    for item in window_info[process_name].values():
        activity_name, timestamp = item
        activity_map[activity_name] = timestamp
    return activity_map

# --- Keyword Extraction to enrich VecDB Query ---
def extract_keywords_from_trace(trace: etree._Element) -> list:
    """Extracts relevant keywords from a single trace in an XES log."""
    keywords = set()
    
    # Trace-level attributes (costCenter, region, etc.)
    for attr in trace.findall("string"):
        val = attr.get("value", "")
        if val and val not in {"UNKNOWN", "MISSING"} and len(val) > 3:
            keywords.add(val)

    # Event-level roles
    for event in trace.findall('event'):
        role = event.find("string[@key='org:role']")
        if role is not None and role.get('value') and role.get('value') not in ["SYSTEM", "MISSING", "UNDEFINED"]:
            keywords.add(role.get('value'))
            
    # Activity stems
    st = PorterStemmer()
    for ev in trace.findall("event"):
        act = ev.find("string[@key='concept:name']").get("value", "")
        for word in re.split(r"[ _]", act):
            if word and word.isalpha() and len(word) > 3:
                keywords.add(st.stem(word.lower()))

    return list(keywords)

def run_drift_agent(state: GraphState) -> dict:
    """
    Parses a user-selected concept drift from a specific or default data directory.
    """
    logging.info("--- Running Drift Agent ---")

    selection = state.get("selected_drift")
    if not selection:
        return {"error": "No drift was selected."}
    
    # --- Intelligent Path Logic ---
    # The agent will use a specific data_dir if provided (for testing),
    # otherwise it will default to the standard drift_outputs folder (for the UI).
    data_dir = selection.get("data_dir")
    if data_dir and Path(data_dir).exists():
        logging.info(f"Using provided data directory: {data_dir}")
        data_dir = Path(data_dir)
    else:
        data_dir = project_root / "data" / "drift_outputs"
        logging.info(f"Using default data directory: {data_dir}")

    try:
        # Assumes there is only one of each file type in the directory
        csv_path = next(data_dir.glob("*.csv"))
        json_path = next(data_dir.glob("*.json"))
        xes_path = next(data_dir.glob("*.xes"))
    except StopIteration:
        return {"error": f"Could not find required .csv, .json, or .xes file in {data_dir}"}

    try:
        drift_df = pd.read_csv(csv_path)
        with open(json_path, 'r') as f:
            window_info = json.load(f)
        log_tree = etree.parse(xes_path)
        traces = log_tree.xpath("//trace")
    except Exception as e:
        return {"error": f"Failed to load or parse data files: {e}"}

    row_index = selection.get("row_index", 0)
    drift_index_in_row = selection.get("drift_index", 0)
    
    if not (0 <= row_index < len(drift_df)):
        return {"error": f"Invalid row index {row_index} for drift data."}
    if not (0 <= row_index < len(traces)):
        return {"error": f"Invalid row index {row_index} for XES traces."}
        
    selected_row = drift_df.iloc[row_index]
    trace_to_analyze = traces[row_index]
    gold_docs = ast.literal_eval(selected_row['gold_source_document'])
    
    logging.info(f"Processing drift #{drift_index_in_row + 1} from CSV row #{row_index + 1}")
    
    # The CSV contains strings like "[('a', 'b')]", we use ast.literal_eval to parse them safely.
    try:
        all_changepoints = ast.literal_eval(selected_row["Detected Changepoints"])
        all_drift_types = ast.literal_eval(selected_row["Detected Drift Types"])
        confidence_str = selected_row["Prediction Confidence"].strip('[]')
        all_confidences = [float(c) for c in confidence_str.split()]

        # Select the specific drift from the lists using its index within the row
        changepoint_pair = all_changepoints[drift_index_in_row]
        drift_type = all_drift_types[drift_index_in_row]
        confidence = all_confidences[drift_index_in_row]
        
        extracted_keywords = extract_keywords_from_trace(trace_to_analyze)

    except (ValueError, SyntaxError, IndexError) as e:
        return {"error": f"Could not parse or index drift data from CSV: {e}"}
    
    # --- 3. Find Timestamps ---
    activity_map = build_activity_to_timestamp_map(window_info)
    start_activity, end_activity = changepoint_pair
    start_timestamp = activity_map.get(start_activity)
    end_timestamp = activity_map.get(end_activity)

    if not all([start_timestamp, end_timestamp]):
        return {"error": f"Could not find timestamps for changepoint pair: {changepoint_pair}"}
    
    # --- 4. Create DriftInfo Object and Update State ---
    drift_info: DriftInfo = {
        "changepoints": changepoint_pair,
        "drift_type": drift_type,
        "confidence": confidence,
        "start_timestamp": start_timestamp,
        "end_timestamp": end_timestamp,
        "gold_doc": gold_docs[drift_index_in_row] # Pass gold doc for logging
    }

    logging.info(f"Populated drift_info: {drift_info}")
    logging.info(f"Extracted Keywords: {extracted_keywords}")
    # '''
    # --- Diagnostic Logging (outcomment if not run) ---
    logging.debug(
        "Keyword list (%d): %s",
        len(extracted_keywords),
        extracted_keywords
    )
    # '''
    # Cap the final list to max 8 items and return or send full list
    # return {"drift_info": drift_info, "drift_keywords": extracted_keywords[:8]}
    return {"drift_info": drift_info, "drift_keywords": extracted_keywords}