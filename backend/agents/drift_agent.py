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

# --- Keyword and Entity Extraction to enrich VecDB Query ---
def extract_keywords_and_entities(trace: etree._Element) -> tuple[list, list]:
    """
    Extracts both general keywords and specific entities from a trace.
    """
    general_keywords = set()
    specific_entities = set()
    st = PorterStemmer()
    
    # Identify specific entities (e.g., project numbers, form IDs)
    for attr in trace.xpath(".//string | .//int | .//float"):
        val = attr.get("value", "")
        # A simple heuristic: if a value contains digits, it's likely an entity
        if val and any(char.isdigit() for char in val) and len(val) > 3:
            specific_entities.add(val.lower())
        # Also capture non-numeric but potentially specific values
        elif val and val not in {"UNKNOWN", "MISSING", "EMPTY"} and len(val) > 3:
            for word in re.split(r'[\s,()-/]', val):
                if word and len(word) > 3:
                    general_keywords.add(st.stem(word.lower()))

    # Extract words from event names
    for event in trace.findall('event'):
        for key in ["concept:name", "activityNameEN"]:
             name_element = event.find(f"string[@key='{key}']")
             if name_element is not None:
                for word in re.split(r'[ _]', name_element.get("value", "")):
                    if word and word.isalpha() and len(word) > 3:
                        general_keywords.add(st.stem(word.lower()))

    return list(general_keywords), list(specific_entities)

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
    
    logging.info(f"Processing drift #{drift_index_in_row + 1} from CSV row #{row_index + 1} ℹ️")
    
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
        
        general_keywords, specific_entities = extract_keywords_and_entities(trace_to_analyze)

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
        "process_name": xes_path.stem, # the .xes file is always named after the process
        "changepoints": changepoint_pair,
        "drift_type": drift_type,
        "confidence": confidence,
        "start_timestamp": start_timestamp,
        "end_timestamp": end_timestamp,
        "gold_doc": gold_docs[drift_index_in_row] # Pass gold doc for logging
    }

    logging.info(f"Populated drift_info: {drift_info}")
    # This provides a clear summary of the keywords and entities extracted.
    logging.info(f"Extracted {len(general_keywords)} general keywords: {general_keywords[:10]}...") # Log first 10
    logging.info(f"Extracted {len(specific_entities)} specific entities: {specific_entities[:10]}...") # Log first 1
    
    # Diagnostic Logging
    # This provides a more detailed view for debugging if needed.
    logging.debug(
        "Keyword list (%d): %s",
        len(general_keywords),
        general_keywords
    )
    logging.debug(
        "Specific Entities (%d): %s",
        len(specific_entities),
        specific_entities
    )

    # Return all data to the state
    return {
        "drift_info": drift_info,
        "drift_keywords": general_keywords,
        "specific_entities": specific_entities
    }