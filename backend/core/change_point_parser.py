import pandas as pd
import json
import ast
import math
import os

# Base folder where CV4CDD outputs are stored
OUTPUT_FOLDER = r"C:\master-thesis\context_drift\data\working_directory_cv4cdd\output_cv4cdd"

CSV_FILENAME = "prediction_results.csv"
JSON_SUBFOLDER = "winsim_images"
JSON_FILENAME = "window_info.json"

# Construct full paths
csv_path = os.path.join(OUTPUT_FOLDER, CSV_FILENAME)
json_path = os.path.join(OUTPUT_FOLDER, JSON_SUBFOLDER, JSON_FILENAME)

def map_change_points_with_dates(csv_path: str, json_path: str) -> pd.DataFrame:
    """
    Maps change point case IDs from CV4CDD output CSV to corresponding event indices, timestamps,
    and includes the associated concept drift type and confidence (rounded up).

    Parameters:
    - csv_path: Path to the prediction_results.csv file
    - json_path: Path to the window_info.json file

    Returns:
    - DataFrame with start/end case IDs, their indices, timestamps, drift types, and confidence
    """
    # Load CSV
    df_csv = pd.read_csv(csv_path)

    # Load JSON
    with open(json_path, "r") as f:
        window_info = json.load(f)

    # Extract mapping from window index to (case_id, timestamp)
    window_mapping = {
        int(index): (entry[0], entry[1])
        for index, entry in window_info["PermitLog"].items()
    }

    # Reverse mapping: from case_id to (index, timestamp)
    reverse_mapping = {
        case_id: (index, timestamp)
        for index, (case_id, timestamp) in window_mapping.items()
    }

    # Parse change points, drift types, and confidence scores
    change_points_list = ast.literal_eval(df_csv["Detected Changepoints"].iloc[0])
    drift_types_list = ast.literal_eval(df_csv["Detected Drift Types"].iloc[0])
    confidence_str = df_csv["Prediction Confidence"].iloc[0].replace(" ", ",")
    confidence_list = ast.literal_eval(confidence_str)

    # Build mapping results
    mapped_change_points = []
    for (start_id, end_id), drift_type, confidence in zip(change_points_list, drift_types_list, confidence_list):
        start_info = reverse_mapping.get(start_id, (None, None))
        end_info = reverse_mapping.get(end_id, (None, None))
        mapped_change_points.append({
            "start_case_id": start_id,
            "start_index": start_info[0],
            "start_timestamp": start_info[1],
            "end_case_id": end_id,
            "end_index": end_info[0],
            "end_timestamp": end_info[1],
            "drift_type": drift_type,
            "confidence (%)": math.ceil(float(confidence) * 100)
        })

    return pd.DataFrame(mapped_change_points)


def format_drifts_for_llm(mapped_df) -> str:
    """
    Converts mapped drift information into a natural-language list for LLM input.
    Includes time range, drift type, and confidence.

    Parameters:
    - mapped_df: DataFrame returned by map_change_points_with_dates()

    Returns:
    - str: Formatted drift descriptions
    """
    lines = []
    for _, row in mapped_df.iterrows():
        line = (
            f"- Between {row['start_timestamp']} and {row['end_timestamp']}, "
            f"a {row['drift_type']} concept drift was detected "
            f"with a confidence of {row['confidence (%)']}%."
        )
        lines.append(line)
    return "\n".join(lines)

# Test output message
"""
if __name__ == "__main__":
    df_mapped = map_change_points_with_dates(csv_path, json_path)
    print(df_mapped.head())

    llm_input = format_drifts_for_llm(df_mapped)
    print("\nDrift Summary for LLM:\n")
    print(llm_input)
"""