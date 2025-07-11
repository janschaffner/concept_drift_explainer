import sys
import pandas as pd
import ast
from pathlib import Path

# --- Path Correction ---
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
# -----------------------

from backend.graph.build_graph import build_graph

def run_master_evaluation():
    """
    Iterates through all test set subdirectories, runs the full pipeline
    for each drift, and consolidates all results into a single report.
    """
    print("--- MASTER EVALUATION HARNESS START ---")

    # 1. Load the golden dataset
    event_logs_dir = project_root / "data" / "event_logs"
    if not event_logs_dir.exists():
        print(f"ERROR: Event logs directory not found at {event_logs_dir}")
        return

    test_set_dirs = [d for d in event_logs_dir.iterdir() if d.is_dir()]
    print(f"Found {len(test_set_dirs)} test sets to evaluate.")

    # 2. Compile the graph once
    app = build_graph()

    results = []

    # 3. Loop through each test set directory
    for test_dir in test_set_dirs:
        test_set_name = test_dir.name
        print(f"\n===== Processing Test Set: {test_set_name} =====")

        try:
            golden_csv_path = next(test_dir.glob("*.csv"))
            df = pd.read_csv(golden_csv_path)
        except (StopIteration, FileNotFoundError):
            print(f"  > WARNING: No prediction_results.csv found in {test_set_name}. Skipping.")
            continue

        # Loop through each drift in the current test set's CSV
        for row_index, row in df.iterrows():
            drift_types = ast.literal_eval(row['Detected Drift Types'])
            gold_docs = ast.literal_eval(row['gold_source_document'])

            for drift_index_in_row, drift_type in enumerate(drift_types):
                drift_id = f"{test_set_name}_drift_{row_index}_{drift_index_in_row}"
                print(f"  Processing {drift_id} ({drift_type})...")

                # UPDATED: Pass the gold_doc into the initial state for better logging
                gold_doc_for_this_drift = gold_docs[drift_index_in_row]
                
                initial_input = {
                    "selected_drift": {
                        "row_index": row_index,
                        "drift_index": drift_index_in_row,
                        "data_dir": str(test_dir),
                        "gold_doc": gold_doc_for_this_drift # Add gold doc for logging
                    }
                }

                final_state = app.invoke(initial_input)

                if final_state.get("error"):
                    print(f"    > ERROR: Pipeline failed. Reason: {final_state['error']}")
                    continue

                explanation = final_state.get("explanation", {})
                ranked_causes = explanation.get("ranked_causes", [])
                gold_doc = gold_doc_for_this_drift.lower()
                cause_docs = [cause.get("source_document", "").lower() for cause in ranked_causes]

                # --- Calculate Metrics ---
                recall_at_1 = 1 if len(cause_docs) > 0 and cause_docs[0] == gold_doc else 0
                recall_at_2 = 1 if gold_doc in cause_docs[:2] else 0

                mrr = 0.0
                try:
                    rank = cause_docs.index(gold_doc) + 1
                    mrr = 1.0 / rank
                except ValueError:
                    mrr = 0.0

                results.append({
                    "test_set": test_set_name,
                    "drift_id": drift_id,
                    "drift_type": drift_type,
                    "gold_document": gold_doc_for_this_drift,
                    "predicted_doc_1": cause_docs[0] if len(cause_docs) > 0 else "N/A",
                    "predicted_doc_2": cause_docs[1] if len(cause_docs) > 1 else "N/A",
                    "recall@1": recall_at_1,
                    "recall@2": recall_at_2,
                    "mrr": mrr
                })

                # --- UPDATED: Print all metrics during the run ---
                print(f"    > Gold Doc: {gold_doc_for_this_drift}")
                print(f"    > Predicted Doc #1: {cause_docs[0] if len(cause_docs) > 0 else 'N/A'}")
                print(f"    > Recall@1: {'HIT ✅' if recall_at_1 else 'MISS ❌'}")
                print(f"    > Recall@2: {'HIT ✅' if recall_at_2 else 'MISS ❌'}")
                print(f"    > Reciprocal Rank: {mrr:.3f}")


    # 4. Save the final report
    if results:
        report_df = pd.DataFrame(results)
        report_path = project_root / "tests" / "master_eval_report.csv"
        report_df.to_csv(report_path, index=False)
        print(f"\n--- MASTER EVALUATION COMPLETE ---")
        print(f"Consolidated evaluation report saved to {report_path}")

        # --- Overall metrics summary ---
        avg_recall_at_1 = report_df['recall@1'].mean()
        avg_recall_at_2 = report_df['recall@2'].mean()
        avg_mrr = report_df['mrr'].mean()

        print("\n--- Overall Metrics ---")
        print(f"Recall@1: {avg_recall_at_1:.2%}")
        print(f"Recall@2: {avg_recall_at_2:.2%}")
        print(f"Mean Reciprocal Rank (MRR): {avg_mrr:.3f}")
        
        # --- NEW: Add assertion for success criterion ---
        # assert avg_recall_at_2 == 1.0, f"Recall@2 failed! Expected 100%, got {avg_recall_at_2:.2%}"
        # print("\n✅ SUCCESS: Recall@2 target met!")

if __name__ == "__main__":
    run_master_evaluation()