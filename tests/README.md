# Purpose of this Directory

This directory contains the scripts for the formal, quantitative evaluation of the Concept Drift Explainer (CDE).

### `run_master_evaluation.py`

This is the **master evaluation harness**. Run this script to perform a full, end-to-end evaluation of the system against all test sets located in `/data/event_logs/`. It will generate a `master_eval_report.csv` file with the final Recall@k and MRR metrics. This script was used to calculate the tests for Eval3.

### `test_single_eval.py`

This is a **single-run utility** for debugging and quick tests. It runs the full pipeline on a single event log, using the data located in the `/data/drift_outputs/` directory.


# Explanation of Evaluation Metrics

The evaluation is based on comparing the system's ranked output against a "golden" dataset where the true root cause document is known for each drift.

**Recall@k**
Recall@k measures whether the correct item appears within the top k results of a ranked list. It is a common metric for evaluating the performance of information retrieval and recommender systems.

Recall@1: This is a strict metric that asks: "Was the single best explanation (the #1 ranked cause) the correct one?" It is a measure of the system's precision and its ability to identify the most likely cause correctly. A score of 100% means the system ranked the golden document as the top cause for every drift.

Recall@2: This is a more lenient metric that asks: "Was the correct explanation present within the top two ranked causes?" This metric is useful because it still considers the system successful if it finds the correct document, even if it ranks another highly plausible cause as #1. A high Recall@2 score indicates that the system is effective at including the true cause in its top recommendations for the analyst.

**Mean Reciprocal Rank (MRR)**
The Mean Reciprocal Rank (MRR) is a statistic that evaluates the quality of a ranked list by considering the rank of the first correct answer.

Calculation: For a single drift, the reciprocal rank is calculated as 1 / rank, where rank is the position of the golden document in the list of causes (1st, 2nd, 3rd, etc.).

If the correct document is ranked 1st, the score is 1/1 = 1.0.

If it is ranked 2nd, the score is 1/2 = 0.5.

If it is ranked 3rd, the score is 1/3 = 0.33.

If the correct document is not found, the rank is considered infinite, and the score is 0.

Interpretation: The final MRR is the average of these reciprocal rank scores across all drifts in the evaluation set. A perfect MRR of 1.0 indicates that the system correctly identified the golden document and placed it at the top of the list for every single case. A higher MRR value indicates that the system is consistently ranking the correct answer closer to the top.