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
    """
    Iterates through all test set subdirectories, runs the full pipeline
    for each drift, and consolidates all results into a single report.
    """
    # --- Create a timestamped log file ---
    log_dir = project_root / "tests" / "logs"
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = log_dir / f"master_eval_log_{timestamp}.txt"
    
    # This initial message will go to the console before logging is configured
    print(f"--- Starting evaluation. Log will be saved to: {log_file_path} ---")

    # --- Setup logging to capture ALL output ---
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    root_logger = logging.getLogger()
    
    # Clean up any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        
    root_logger.setLevel(logging.INFO)

    # Add a handler to write everything to the log file with UTF-8 encoding
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)
    
    # FIX: Add a handler to also stream the detailed log to the console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)
    
    # Suppress Pydantic V1 warnings
    warnings.filterwarnings("ignore", message=".*Pydantic BaseModel V1.*")

    # --- Start of Script Logic ---
    logging.info("--- MASTER EVALUATION HARNESS START ---")

    event_logs_dir = project_root / "data" / "event_logs"
    if not event_logs_dir.exists():
        logging.error(f"ERROR: Event logs directory not found at {event_logs_dir}")
        return

    test_set_dirs = [d for d in event_logs_dir.iterdir() if d.is_dir()]
    logging.info(f"Found {len(test_set_dirs)} test sets to evaluate.")

    app = build_graph()
    results = []

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

        for row_index, row in df.iterrows():
            drift_types = ast.literal_eval(row['Detected Drift Types'])
            gold_docs = ast.literal_eval(row['gold_source_document'])

            for drift_index_in_row, drift_type in enumerate(drift_types):
                drift_id = f"{test_set_name}_drift_{row_index}_{drift_index_in_row}"
                logging.info(f"  Processing {drift_id} ({drift_type})...")

                gold_doc_for_this_drift = gold_docs[drift_index_in_row]
                
                initial_input = {
                    "selected_drift": {
                        "row_index": row_index, "drift_index": drift_index_in_row,
                        "data_dir": str(test_dir), "gold_doc": gold_doc_for_this_drift
                    }
                }

                final_state = app.invoke(initial_input)

                if final_state.get("error"):
                    logging.error(f"    > ERROR: Pipeline failed. Reason: {final_state['error']}")
                    continue

                # Add the successful state to our list for meta-analysis
                states_for_this_log.append(final_state)

                explanation = final_state.get("explanation", {})
                ranked_causes = explanation.get("ranked_causes", [])
                
                gold_doc_stem = Path(gold_doc_for_this_drift).stem.lower()
                cause_doc_stems = [Path(c.get("source_document", "")).stem.lower() for c in ranked_causes]

                recall_at_1 = 1 if len(cause_doc_stems) > 0 and cause_doc_stems[0] == gold_doc_stem else 0
                recall_at_2 = 1 if gold_doc_stem in cause_doc_stems[:2] else 0

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

        # Run the Drift Linker Agent on the results from the current event log
        if len(states_for_this_log) >= 2:
            logging.info(f"\n===== Running Meta-Analysis for {test_set_name} =====")
            linker_result = run_drift_linker_agent(states_for_this_log)
            if linker_result.get("error"):
                logging.error(f"Drift Linker Agent failed: {linker_result['error']}")
            elif linker_result.get("linked_drift_summary"):
                logging.info("--- Drift Linker Agent Summary ---")
                logging.info(f"  Connection Type: {linker_result.get('connection_type')}")
                logging.info(f"  Summary: {linker_result.get('linked_drift_summary')}")

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