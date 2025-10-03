"""
This module serves as the master evaluation harness for the CDE project.

It is designed to be run as a standalone script to perform a full, end-to-end
evaluation of the agentic pipeline against the curated "gold standard" test
set. The script iterates through all test event logs, runs the complete CDE
pipeline for each detected drift, and calculates key performance metrics,
including Recall@1, Recall@2, and Mean Reciprocal Rank (MRR).

The final output is a timestamped log file with detailed execution traces and a
`master_eval_report.csv` file that consolidates the performance metrics for
all test cases.
"""

import sys
import pandas as pd
import ast
from pathlib import Path
import logging
from datetime import datetime
import warnings

# --- Path Correction ---
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
# -----------------------

from backend.graph.build_graph import build_graph
from backend.agents.drift_linker_agent import run_drift_linker_agent

def run_master_evaluation():
    """Runs the full end-to-end evaluation process.

    This function orchestrates the entire evaluation by:
    1. Setting up a detailed, timestamped logging system.
    2. Discovering and iterating through all test set directories.
    3. For each drift in the test set, invoking the full LangGraph pipeline.
    4. Calculating performance metrics (Recall@k, MRR) by comparing the
       pipeline's output to the known "gold document".
    5. Optionally running the Drift Linker Agent for multi-drift logs.
    6. Generating a final CSV report with all results.
    """
    # --- Step 1: Setup Logging ---
    # Create a unique, timestamped log file for this evaluation run.
    # This captures a complete trace of the agent's reasoning for later analysis.
    log_dir = project_root / "tests" / "logs"
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = log_dir / f"master_eval_log_{timestamp}.txt"
    
    # This initial message will go to the console before logging is configured
    print(f"--- Starting evaluation. Log will be saved to: {log_file_path} ---")

    # Configure the root logger to capture all output from all modules.
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    root_logger = logging.getLogger()
    
    # Clean up any existing handlers to prevent duplicate logging.
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        
    root_logger.setLevel(logging.INFO)

    # Add a handler to write everything to the log file with UTF-8 encoding
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)
    
    # Add a handler to also stream the detailed log to the console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)
    
    # Suppress Pydantic V1 warnings
    warnings.filterwarnings("ignore", message=".*Pydantic BaseModel V1.*")

    # --- Step 2: Discover Test Sets and Initialize the Graph ---
    logging.info("--- MASTER EVALUATION HARNESS START ---")

    event_logs_dir = project_root / "data" / "event_logs"
    if not event_logs_dir.exists():
        logging.error(f"ERROR: Event logs directory not found at {event_logs_dir}")
        return

    test_set_dirs = [d for d in event_logs_dir.iterdir() if d.is_dir()]
    logging.info(f"Found {len(test_set_dirs)} test sets to evaluate.")

    # Build the compiled LangGraph application once.
    app = build_graph()
    results = []

    # --- Step 3: Main Evaluation Loop ---
    # Iterate through each test set directory (representing one event log).
    for test_dir in test_set_dirs:
        test_set_name = test_dir.name
        logging.info(f"\n===== Processing Test Set: {test_set_name} =====")
        # Reset the list for each new event log to ensure separate analysis
        states_for_this_log = []

        try:
            golden_csv_path = next(test_dir.glob("*.csv"))
            df = pd.read_csv(golden_csv_path)
        except (StopIteration, FileNotFoundError):
            logging.warning(f"  > WARNING: No prediction_results.csv found in {test_set_name}. Skipping.")
            continue

        # Nested loop: Iterate through each row and each drift within that row.
        for row_index, row in df.iterrows():
            drift_types = ast.literal_eval(row['Detected Drift Types'])
            gold_docs = ast.literal_eval(row['gold_source_document'])

            for drift_index_in_row, drift_type in enumerate(drift_types):
                drift_id = f"{test_set_name}_drift_{row_index}_{drift_index_in_row}"
                logging.info(f"  Processing {drift_id} ({drift_type})...")

                gold_doc_for_this_drift = gold_docs[drift_index_in_row]
                
                # Prepare the initial input for the graph, including the gold document
                # for evaluation purposes.
                initial_input = {
                    "selected_drift": {
                        "row_index": row_index, "drift_index": drift_index_in_row,
                        "data_dir": str(test_dir), "gold_doc": gold_doc_for_this_drift
                    }
                }

                # Invoke the full agentic pipeline for this single drift.
                final_state = app.invoke(initial_input)

                if final_state.get("error"):
                    logging.error(f"    > ERROR: Pipeline failed. Reason: {final_state['error']}")
                    continue

                # Add the successful state to our list for meta-analysis
                states_for_this_log.append(final_state)

                # --- Step 4: Calculate Performance Metrics ---
                # Compare the ranked results against the known gold document.
                explanation = final_state.get("explanation", {})
                ranked_causes = explanation.get("ranked_causes", [])
                
                gold_doc_stem = Path(gold_doc_for_this_drift).stem.lower()
                cause_doc_stems = [Path(c.get("source_document", "")).stem.lower() for c in ranked_causes]

                # Recall@1: Is the top-ranked document the correct one?
                recall_at_1 = 1 if len(cause_doc_stems) > 0 and cause_doc_stems[0] == gold_doc_stem else 0
                # Recall@2: Is the correct document in the top two positions?
                recall_at_2 = 1 if gold_doc_stem in cause_doc_stems[:2] else 0

                # MRR: Calculate the reciprocal rank of the first correct answer.
                mrr = 0.0
                try:
                    rank = cause_doc_stems.index(gold_doc_stem) + 1
                    mrr = 1.0 / rank
                except ValueError:
                    mrr = 0.0

                results.append({
                    "test_set": test_set_name, "drift_id": drift_id, "drift_type": drift_type,
                    "gold_document": gold_doc_for_this_drift,
                    "predicted_doc_1": ranked_causes[0].get("source_document") if len(ranked_causes) > 0 else "N/A",
                    "predicted_doc_2": ranked_causes[1].get("source_document") if len(ranked_causes) > 1 else "N/A",
                    "recall@1": recall_at_1, "recall@2": recall_at_2, "mrr": mrr
                })

                logging.info(f"    > Gold Doc: {gold_doc_for_this_drift}")
                logging.info(f"    > Predicted Doc #1: {ranked_causes[0].get('source_document') if len(ranked_causes) > 0 else 'N/A'}")
                logging.info(f"    > Recall@1: {'HIT ✅' if recall_at_1 else 'MISS ❌'}")
                logging.info(f"    > Recall@2: {'HIT ✅' if recall_at_2 else 'MISS ❌'}")
                logging.info(f"    > Reciprocal Rank: {mrr:.3f}")

        # --- Step 5: Run Meta-Analysis (Drift Linker) ---
        # If the log had multiple drifts, run the Drift Linker agent on the collected states.
        if len(states_for_this_log) >= 2:
            logging.info(f"\n===== Running Meta-Analysis for {test_set_name} =====")
            linker_result = run_drift_linker_agent(states_for_this_log)
            if linker_result.get("error"):
                logging.error(f"Drift Linker Agent failed: {linker_result['error']}")
            elif linker_result.get("linked_drift_summary"):
                logging.info("--- Drift Linker Agent Summary ---")
                logging.info(f"  Connection Type: {linker_result.get('connection_type')}")
                logging.info(f"  Summary: {linker_result.get('linked_drift_summary')}")

    # --- Step 6: Generate Final Report ---
    # Consolidate all results into a single CSV and print a summary of the overall metrics.
    if results:
        report_df = pd.DataFrame(results)
        report_path = project_root / "tests" / "master_eval_report.csv"
        report_df.to_csv(report_path, index=False)
        
        avg_recall_at_1 = report_df['recall@1'].mean()
        avg_recall_at_2 = report_df['recall@2'].mean()
        avg_mrr = report_df['mrr'].mean()

        summary = (
            f"\n--- MASTER EVALUATION COMPLETE ---\n"
            f"Consolidated evaluation report saved to {report_path}\n\n"
            f"--- Overall Metrics ---\n"
            f"Recall@1: {avg_recall_at_1:.2%}\n"
            f"Recall@2: {avg_recall_at_2:.2%}\n"
            f"Mean Reciprocal Rank (MRR): {avg_mrr:.3f}\n"
        )
        
        logging.info(summary)

if __name__ == "__main__":
    run_master_evaluation()