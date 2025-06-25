import pandas as pd
import json
import ast  # Safely evaluate string representations of Python literals
from pathlib import Path
import logging

# Import our graph state schema
from ..state.schema import GraphState, DriftInfo

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def build_activity_to_timestamp_map(window_info: dict) -> dict:
    """
    Pre-processes the window_info.json data into a direct lookup table for efficiency.
    Maps activity_name -> timestamp.
    
    Args:
        window_info: The loaded content of window_info.json.

    Returns:
        A dictionary mapping each activity name to its timestamp.
    """
    activity_map = {}
    # The first key is the process name, e.g., name of the XES file (the Event Log)
    process_name = next(iter(window_info))
    for item in window_info[process_name].values():
        activity_name, timestamp = item
        activity_map[activity_name] = timestamp
    return activity_map


def run_drift_agent(state: GraphState) -> dict:
    """
    Parses the concept drift detection results and enriches the graph state.

    This agent reads the output from the CV4CDD framework, identifies the
    first detected drift, finds its corresponding timestamps, and populates
    the `drift_info` field in the state.

    Args:
        state: The current graph state.

    Returns:
        A dictionary with the updated `drift_info` field for the state.
    """
    logging.info("--- Running Drift Agent ---")
    
    # --- 1. Define File Paths ---
    # Construct paths relative to a project root.
    # We assume the script is run from the 'context_drift_explainer' root directory.
    base_data_path = Path("data/drift_outputs")
    csv_path = base_data_path / "prediction_results.csv"
    json_path = base_data_path / "winsim_images" / "window_info.json"

    logging.info(f"Loading drift data from: {csv_path}")
    logging.info(f"Loading timestamp data from: {json_path}")

    # --- 2. Load and Parse Data ---
    try:
        drift_df = pd.read_csv(csv_path)
        with open(json_path, 'r') as f:
            window_info = json.load(f)
    except FileNotFoundError as e:
        logging.error(f"Error loading files: {e}. Make sure paths are correct.")
        # Return a partial state to indicate failure
        return {"error": str(e)}

    # For this implementation, we process only the FIRST drift instance in the file.
    # The architecture can be extended to handle multiple drifts in a loop.                         --> TODO: Design Objective 9
    if drift_df.empty:
        logging.warning("Prediction results file is empty. No drift to process.")
        return {"error": "Prediction results file is empty."}
        
    first_drift_row = drift_df.iloc[0]
    
    # The CSV contains strings like "[('a', 'b')]", we use ast.literal_eval to parse them safely.
    try:
        all_changepoints = ast.literal_eval(first_drift_row["Detected Changepoints"])
        all_drift_types = ast.literal_eval(first_drift_row["Detected Drift Types"])
        # The confidence string looks like "[0.7609 0.6718]", requires special handling --> differentiate for multiple drifts
        confidence_str = first_drift_row["Prediction Confidence"].strip('[]')
        all_confidences = [float(c) for c in confidence_str.split()]
    except (ValueError, SyntaxError) as e:
        logging.error(f"Error parsing data from CSV row: {e}")
        return {"error": f"Could not parse CSV content: {e}"}

    changepoint_pair = all_changepoints[0]
    drift_type = all_drift_types[0]
    confidence = all_confidences[0]

    # --- 3. Find Timestamps ---
    activity_map = build_activity_to_timestamp_map(window_info)
    
    start_activity, end_activity = changepoint_pair
    start_timestamp = activity_map.get(start_activity, None)
    end_timestamp = activity_map.get(end_activity, None)

    if not all([start_timestamp, end_timestamp]):
        error_msg = f"Could not find timestamps for changepoint pair: {changepoint_pair}"
        logging.error(error_msg)
        return {"error": error_msg}

    # --- 4. Create DriftInfo Object and Update State ---
    drift_info: DriftInfo = {
        "changepoints": changepoint_pair,
        "drift_type": drift_type,
        "confidence": confidence,
        "start_timestamp": start_timestamp,
        "end_timestamp": end_timestamp,
    }

    logging.info("Drift Agent execution successful.")
    logging.info(f"Populated drift_info: {drift_info}")

    return {"drift_info": drift_info}