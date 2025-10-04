# Purpose of this Directory

This directory serves as the **master archive** for all evaluation data used by the CDE.

It contains the original, "gold standard" versions of the CV4CDD-4D output files for each event log, including the crucial `prediction_results.csv` file that has been augmented with the `gold_source_document` column.

## Workflow

- **To run the full evaluation:** Copy the contents of the desired subdirectories from here into the `/data/event_logs/` directory. The `run_master_evaluation.py` script will then use them as input (Note: already done, no need to copy).
- **To analyze a single event log in the UI:** Copy the three required files (`.csv`, `.json`, `.xes`) from the relevant subdirectory here into the `/data/drift_outputs/` directory. This will allow the Streamlit app to use it as a fallback data source.

**Note:** This folder should be treated as the pristine "source of truth" for evaluation data. Do not edit the files here directly; always work with copies.