# Purpose of this Directory

**This directory serves as a default or fallback location for the raw drift detection data.**

The primary workflow of the CDE is designed to receive the path to the drift data directly from the frontend UI or from an evaluation script.

This folder is used by the `Drift Agent` only if that primary path is not provided or is invalid. 
The folder is also used by the `test_single_eval.py` to perform single tests on one event log set.
It is primarily intended for local, standalone testing and to serve as a structural example of how the drift output data should be organized.