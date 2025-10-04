# Launch the Streamlit interface

```bash
streamlit run frontend/app.py
```

Place PDFs, PPTX, DOCX/TXT, PNG, or JPG files in `frontend/static/documents/`. Filenames must start with a date formatted as `YYYY-MM-DD_` (e.g., `2017-10-11_Conference_Guidelines.pdf`). Files without a parsable date are skipped. The synthetic context document corpus used for evaluation is already stored here.

Event logs must be placed in `data/event_logs/`(already done).