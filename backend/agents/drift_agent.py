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

# Configure basic logging.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
MODEL_NAME = "gpt-4o-mini"

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
    st = PorterStemmer()
    
    if trace is None:
        return []

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

def _find_trace_by_id(traces: list, changepoint_id: str) -> etree._Element:
    """
    Finds a specific trace by matching the numeric ID from the changepoint.
    """
    # --- CHANGE START ---
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
    # --- CHANGE END ---

def _format_trace_for_llm(trace: etree._Element) -> str:
    """Formats a trace's events into a clean, human-readable string for the LLM."""
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
        with open(json_path, 'r', encoding='utf-8') as f: window_info = json.load(f)
        log_tree = etree.parse(xes_path)
        traces = log_tree.xpath("//trace")
    except (StopIteration, Exception) as e:
        return {"error": f"Failed to load or find data files in {data_dir}: {e}"}

    row_index = selection.get("row_index", 0)
    drift_index_in_row = selection.get("drift_index", 0)
    selected_row = drift_df.iloc[row_index]
    
    logging.info(f"ℹ️  Processing drift #{drift_index_in_row + 1} from CSV row #{row_index + 1}  ℹ️")
    
    try:
        all_changepoints = ast.literal_eval(selected_row["Detected Changepoints"])
        changepoint_pair = all_changepoints[drift_index_in_row]
        logging.debug(f"Looking for timestamps for: {changepoint_pair}")

        start_trace = _find_trace_by_id(traces, changepoint_pair[0])
        end_trace = _find_trace_by_id(traces, changepoint_pair[1])
        logging.debug(f"DEBUGGING {start_trace}")
        logging.debug(f"DEBUGGING {end_trace}")

        if start_trace is None or end_trace is None:
            return {"error": f"Could not find one or both traces for changepoints: {changepoint_pair}"}
            
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
            logging.warning(f"LLM summary for drift phrase failed: {e}. Falling back to basic phrase.")
            summary = " ".join(re.findall(r"[A-Za-z]+", " ".join(changepoint_pair))).lower()

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