"""
This module contains the implementation of the Drift Agent, the first agent in the
Concept Drift Explainer's pipeline. Its primary responsibility is to take the
raw output from a concept drift detector (CV4CDD-4D), parse it, and transform
it into a structured, semantically rich representation of the drift that can be
used by downstream agents for retrieval and explanation.
"""

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

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from backend.state.schema import GraphState, DriftInfo
from backend.utils.cache import load_cache, save_to_cache, get_cache_key

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
MODEL_NAME = "gpt-4o-mini"

# --- Helper Functions ---

def build_activity_to_timestamp_map(window_info: dict) -> dict:
    """Converts the raw window_info.json data into a simple lookup map.

    This pre-processes the nested JSON structure into a flat dictionary that maps
    an activity instance ID directly to its ISO 8601 timestamp string, allowing for
    efficient lookups.

    Args:
        window_info: The loaded JSON data from the detector's output.

    Returns:
        A dictionary mapping activity IDs to their timestamps.
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
    """Extracts and normalizes a flat list of keywords from an XES trace element.

    This function iterates through all string, int, and float attributes within a
    trace, as well as the 'concept:name' of each event, to build a set of
    relevant keywords. It applies several heuristics to filter out noise, such
    as removing numeric IDs and short words, and uses a Porter Stemmer to
    normalize the keywords to their root form.

    Args:
        trace: An lxml etree._Element representing a single <trace>.

    Returns:
        A list of unique, stemmed keywords.
    """
    general_keywords = set()
    st = PorterStemmer()
    
    if trace is None:
        return []

    # First pass: Extract keywords from all general trace attributes.
    for attr in trace.xpath(".//string | .//int | .//float"):
        val = attr.get("value", "")
        # Filter out common noise and purely numeric values.
        if val and val not in {"UNKNOWN", "MISSING", "EMPTY"} and not any(char.isdigit() for char in val) and len(val) > 3:
            for word in re.split(r'[\s,()/-]', val):
                if word and len(word) > 3:
                    general_keywords.add(st.stem(word.lower()))

    # Second pass: Extract keywords from the 'concept:name' of each event.
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

def _find_trace_by_id(traces: list, changepoint_id: str) -> etree._Element:
    """Internal helper to find a specific trace element by its numeric ID.

    The CV4CDD-4D detector output uses activity instance IDs (e.g.,
    "declaration 81722"). This function robustly extracts the numeric part of
    that ID and searches the full .xes log to find the corresponding <trace>
    element whose 'concept:name' contains the same numeric ID.

    Args:
        traces: A list of all <trace> elements from the .xes file.
        changepoint_id: The activity instance ID from the detector's output.

    Returns:
        The matching lxml etree._Element for the trace, or None if not found.
    """
    logging.info(f"Attempting to find trace for changepoint ID: '{changepoint_id}'")
    changepoint_nums = re.findall(r'\d+', changepoint_id)
    if not changepoint_nums:
        logging.warning(f"Could not extract a numeric ID from changepoint '{changepoint_id}'.")
        return None
    target_num = changepoint_nums[-1]
    logging.info(f"Searching for trace with numeric ID: {target_num}")

    for trace in traces:
        name_element = trace.find("string[@key='concept:name']")
        if name_element is not None:
            trace_name = name_element.get("value", "")
            logging.debug(f"  ...checking against trace: '{trace_name}'")
            trace_nums = re.findall(r'\d+', trace_name)
            if trace_nums and trace_nums[-1] == target_num:
                logging.info(f"✅ Match found for ID {target_num}: '{trace_name}'")
                return trace
    
    logging.warning(f"❌ No matching trace found for numeric ID {target_num}.")
    return None

def _format_trace_for_llm(trace: etree._Element) -> str:
    """Internal helper to format a trace's events into a clean string for an LLM.

    Args:
        trace: The lxml etree._Element representing the trace.

    Returns:
        A formatted, human-readable string of the trace's events.
    """
    if trace is None:
        return "Trace not found."
    
    events_str = []
    for event in trace.findall('event'):
        name_element = event.find("string[@key='concept:name']")
        time_element = event.find("date[@key='time:timestamp']")
        
        if name_element is not None and time_element is not None:
            name = name_element.get("value", "N/A")
            timestamp = time_element.get("value", "N/A")
            events_str.append(f"- [{timestamp}] {name}")
            
    return "\n".join(events_str)

# --- Main Agent Logic ---

def run_drift_agent(state: GraphState) -> dict:
    """The entrypoint agent for the CDE pipeline.

    This agent performs the initial data ingestion and transformation. It reads
    the raw output files from the CV4CDD-4D detector, identifies the specific
    drift selected by the user, and performs a comparative trace analysis using
    an LLM to generate a rich, semantic `drift_phrase`. It populates the initial
    `GraphState` with the `drift_info`, `drift_keywords`, and `drift_phrase`,
    which serve as the foundation for all downstream agents.

    Args:
        state: The current graph state. Must contain the `selected_drift` key.

    Returns:
        A dictionary with the updated state fields to be merged into the `GraphState`.
    """
    logging.info("--- Running Drift Agent ---")
    selection = state.get("selected_drift")
    if not selection:
        return {"error": "No drift was selected."}
    
    # Step 1: Load all necessary data files (CSV, JSON, XES) from the frontend, 
    # else from the detector output directory (e.g., for testing).
    data_dir = selection.get("data_dir")
    data_dir = Path(data_dir) if data_dir and Path(data_dir).exists() else project_root / "data" / "drift_outputs"
    logging.info(f"Using data directory: {data_dir}")

    try:
        csv_path = next(data_dir.glob("*.csv"))
        json_path = next(data_dir.glob("*.json"))
        xes_path = next(data_dir.glob("*.xes"))
        
        drift_df = pd.read_csv(csv_path)
        with open(json_path, 'r', encoding='utf-8') as f: window_info = json.load(f)
        log_tree = etree.parse(xes_path)
        traces = log_tree.xpath("//trace")
    except (StopIteration, Exception) as e:
        return {"error": f"Failed to load or find data files in {data_dir}: {e}"}

    # Step 2: Isolate the specific drift to be analyzed based on the user's selection from the UI.
    row_index = selection.get("row_index", 0)
    drift_index_in_row = selection.get("drift_index", 0)
    selected_row = drift_df.iloc[row_index]
    
    logging.info(f"ℹ️  Processing drift #{drift_index_in_row + 1} from CSV row #{row_index + 1}  ℹ️")
    
    try:
        all_changepoints = ast.literal_eval(selected_row["Detected Changepoints"])
        changepoint_pair = all_changepoints[drift_index_in_row]
        logging.debug(f"Looking for timestamps for: {changepoint_pair}")

        # Find the full "before" and "after" traces from the .xes log.
        start_trace = _find_trace_by_id(traces, changepoint_pair[0])
        end_trace = _find_trace_by_id(traces, changepoint_pair[1])
        logging.debug(f"DEBUGGING {start_trace}")
        logging.debug(f"DEBUGGING {end_trace}")

        if start_trace is None or end_trace is None:
            return {"error": f"Could not find one or both traces for changepoints: {changepoint_pair}"}

        # Extract keywords for the broad, initial retrieval query.    
        start_keywords = extract_general_keywords(start_trace)
        end_keywords = extract_general_keywords(end_trace)
        general_keywords = list(set(start_keywords + end_keywords))
            
        all_drift_types = ast.literal_eval(selected_row["Detected Drift Types"])
        drift_type = all_drift_types[drift_index_in_row]
        
        # Robustly parse confidence scores, whether they are comma or space-separated
        raw_confidence = selected_row["Prediction Confidence"].strip('[]').replace(',', ' ')
        all_confidences = [float(c) for c in raw_confidence.split() if c]

        confidence = all_confidences[drift_index_in_row]
    except (ValueError, SyntaxError, IndexError) as e:
        return {"error": f"Could not parse or index drift data from CSV: {e}"}
    
    # Step 3: Perform comparative trace analysis with an LLM to generate a rich, semantic drift_phrase.
    load_dotenv()
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
    llm_cache = load_cache()
    
    formatted_start_trace = _format_trace_for_llm(start_trace)
    formatted_end_trace = _format_trace_for_llm(end_trace)

    prompt_template = """You are a concise business process analyst specializing in travel expense claims. Your task is to summarize the change between two process traces.

**Overall Process Context:**
The process involves travel expense claims at a university. There are two main trip types: domestic (no prior permission needed) and international (requires a pre-approved travel permit). The general flow is: Employee Submission -> Administration -> Budget Owner -> Supervisor -> Payment.

**Analysis Task:**
1. Give a one sentence description of the process context. 
2. Take the two snapshots of a process instance below, one from the beginning of a drift period and one from the end. Compare the two and summarize the key change in the process flow in one single, descriptive sentence.

**Start Point Trace:**
{start_trace}

**End Point Trace:**
{end_trace}

**Two-sentence summary of the change:**
"""
    prompt = prompt_template.format(start_trace=formatted_start_trace, end_trace=formatted_end_trace)
    cache_key = get_cache_key(prompt, MODEL_NAME)
    
    summary = llm_cache.get(cache_key)
    if not summary:
        logging.info("CACHE MISS. Calling API for drift phrase summary...")
        try:
            response = llm.invoke(prompt)
            summary = response.content.strip()
            llm_cache[cache_key] = summary
            save_to_cache(llm_cache)
        except Exception as e:
            # Fallback to a simple phrase if the LLM call fails.
            logging.warning(f"LLM summary for drift phrase failed: {e}. Falling back to basic phrase.")
            summary = " ".join(re.findall(r"[A-Za-z]+", " ".join(changepoint_pair))).lower()

    # Step 4: Assemble and return the initial state fields for the graph.
    process_name = xes_path.stem
    drift_phrase = f"{process_name}: {summary}"
    
    activity_map = build_activity_to_timestamp_map(window_info)
    start_timestamp = activity_map.get(changepoint_pair[0])
    end_timestamp = activity_map.get(changepoint_pair[1])

    drift_info: DriftInfo = {
        "process_name": process_name, 
        "changepoints": changepoint_pair,
        "drift_type": drift_type, 
        "confidence": confidence,
        "start_timestamp": start_timestamp, 
        "end_timestamp": end_timestamp,
    }
    
    # Pass along the gold document if it's provided (for evaluation).
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