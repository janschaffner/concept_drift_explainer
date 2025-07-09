import sys
import pandas as pd
import ast
from pathlib import Path

# --- Path Correction ---
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
# -----------------------

from backend.graph.build_graph import build_graph

def run_evaluation():
    """
    Runs the full pipeline for each drift in the golden dataset
    and calculates evaluation metrics.
    """
    print("---  EVALUATION HARNESS START ---")
    
    # 1. Load the golden dataset
    golden_csv_path = project_root / "data" / "drift_outputs" / "prediction_results.csv"
    if not golden_csv_path.exists():
        print(f"ERROR: Golden CSV not found at {golden_csv_path}")
        return
        
    df = pd.read_csv(golden_csv_path)
    print(f"Found {len(df)} rows in the golden dataset.")

    # 2. Compile the graph once
    app = build_graph()
    
    results = []
    
    # 3. Loop through each drift and run the analysis
    for row_index, row in df.iterrows():
        drift_types = ast.literal_eval(row['Detected Drift Types'])
        gold_docs = ast.literal_eval(row['gold_source_document'])
        
        for drift_index_in_row, drift_type in enumerate(drift_types):
            print(f"\nProcessing Drift: row {row_index}, index {drift_index_in_row} ({drift_type})...")
            
            # Prepare input for the graph
            initial_input = {"selected_drift": {"row_index": row_index, "drift_index": drift_index_in_row}}
            
            # Invoke the full pipeline
            final_state = app.invoke(initial_input)
            
            if final_state.get("error"):
                print(f"  > ERROR: Pipeline failed for this drift. Reason: {final_state['error']}")
                continue

            # Extract results for evaluation
            explanation = final_state.get("explanation", {})
            ranked_causes = explanation.get("ranked_causes", [])
            gold_doc = gold_docs[drift_index_in_row]
            
            # Get the list of source documents from the ranked causes
            cause_docs = [cause.get("source_document", "").lower() for cause in ranked_causes]
            gold_doc_lower = gold_doc.lower()
            
            # --- Calculate Metrics ---
            
            # Recall@1
            recall_at_1 = 1 if len(cause_docs) > 0 and cause_docs[0] == gold_doc_lower else 0
            
            # Recall@2
            recall_at_2 = 1 if gold_doc_lower in cause_docs[:2] else 0

            # Mean Reciprocal Rank (MRR)
            mrr = 0.0
            try:
                rank = cause_docs.index(gold_doc_lower) + 1
                mrr = 1.0 / rank
            except ValueError:
                mrr = 0.0 # Gold document was not found

            results.append({
                "row": row_index,
                "drift_index": drift_index_in_row,
                "drift_type": drift_type,
                "gold_document": gold_doc,
                "recall@1": recall_at_1,
                "recall@2": recall_at_2,
                "mrr": mrr
            })
            
            print(f"  > Gold Doc: {gold_doc}")
            print(f"  > Top 2 Predicted Docs: {[cause.get('source_document') for cause in ranked_causes[:2]]}")
            print(f"  > Recall@2: {'HIT' if recall_at_2 else 'MISS'}")


    # 4. Save the final report
    if results:
        report_df = pd.DataFrame(results)
        report_path = project_root / "tests" / "eval_report.csv"
        report_df.to_csv(report_path, index=False)
        print(f"\n--- EVALUATION COMPLETE ---")
        print(f"Evaluation report saved to {report_path}")
        
        # Print summary metrics
        avg_recall_at_1 = report_df['recall@1'].mean()
        avg_recall_at_2 = report_df['recall@2'].mean()
        avg_mrr = report_df['mrr'].mean()
        
        print("\n--- Overall Metrics ---")
        print(f"Recall@1: {avg_recall_at_1:.2%}")
        print(f"Recall@2: {avg_recall_at_2:.2%}")
        print(f"Mean Reciprocal Rank (MRR): {avg_mrr:.3f}")


if __name__ == "__main__":
    run_evaluation()