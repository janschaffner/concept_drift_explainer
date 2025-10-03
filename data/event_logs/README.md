# Purpose of this Directory

This directory contains the **primary input data for the master evaluation harness** (`/tests/run_master_evaluation.py`).

Each subdirectory within this folder represents a single test set, corresponding to one event log. Each of these subdirectories must contain the output files from the CV4CDD-4D detector, including the crucial `.csv` file which has been augmented with the `gold_source_document` column to serve as the ground truth for evaluation.

The structure of this directory is critical for the evaluation script to run correctly.