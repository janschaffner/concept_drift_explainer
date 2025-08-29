## Concept Drift Explainer

A prototype application for explaining concept drift in business processes. It links detected drift periods from event logs with relevant internal and external context (e.g., policy changes, organizational memos, presentations) to produce concise, evidence‑backed explanations of what changed and why.

The research question driving this work: How can internal and external context dimensions be integrated into process mining sense‑making?

Built with a multi‑agent pipeline (LangGraph) and a simple Streamlit UI, the system ingests context documents into a vector database, retrieves and re‑ranks relevant evidence for a given drift, maps evidence to a BPM context taxonomy, and synthesizes an explanation with citations and confidence scores.

---

## Features

- Automated explanations: Generates a structured explanation with summary and ranked, evidence‑backed causes.
- Multimodal ingestion: Reads PDFs, PPTX, DOCX/TXT, and images; uses GPT‑4o vision for images in slides.
- Agentic pipeline: Orchestrated via LangGraph with specialized agents (retrieval, reranking, mapping, synthesis, chatbot).
- Hybrid vector search: Pinecone with two namespaces — `context` for documents and `bpm-kb` for a BPM glossary.
- Interactive UI: Streamlit app to pick event logs, run analysis, inspect causes, export DOCX, and chat.
- Evaluation assets: Test logs and scripts to reproduce end‑to‑end runs.

---

## Repository Structure

- `frontend/app.py`: Streamlit application entry point.
- `frontend/pages/`: UI pages (Home, Manage Context, Settings).
- `frontend/static/documents/`: Context documents for ingestion (PDF, PPTX, DOCX, TXT, PNG/JPG).
- `backend/graph/build_graph.py`: Assembles the LangGraph workflow.
- `backend/agents/`: Drift analysis, retrieval, re‑ranking, mapping, explanation, and chatbot agents.
- `backend/state/schema.py`: TypedDict state schema passed through the graph.
- `backend/utils/ingest_documents.py`: Ingestion pipeline for documents and glossary (creates/connects Pinecone index).
- `backend/utils/reporting.py`: DOCX export of explanations.
- `data/event_logs/<LogName>/`: Event logs with `*.csv`, `*.json`, `*.xes` per log.
- `data/knowledge_base/bpm_glossary.csv`: BPM glossary ingested into `bpm-kb` namespace.
- `tests/`: Evaluation assets and scripts.

---

## Architecture (High Level)

1) Ingestion
- Extracts text (and image descriptions) from `frontend/static/documents/`.
- Embeds chunks with `text-embedding-3-small` (1536‑D) and upserts into Pinecone `context` namespace.
- Embeds `data/knowledge_base/bpm_glossary.csv` into Pinecone `bpm-kb` namespace.

2) Explanation Pipeline (LangGraph)
- Drift Agent: Parses selected drift window from `*.csv/json/xes`, generates keywords and a drift phrase.
- Context Retrieval Agent: Hybrid semantic + temporal retrieval from Pinecone (`context` + `bpm-kb`).
- Re‑Ranker Agent: Curates and scores snippets; keeps a safety net candidate.
- Franzoi Mapper Agent: Maps evidence to Franzoi et al. context taxonomy paths.
- Explanation Agent: Synthesizes a summary and ranked causes; calibrates confidence.
- Chatbot Agent: Answers follow‑ups using the full state log.

---

## Prerequisites

- Python 3.10+ (3.11 recommended)
- OpenAI API key (for embeddings, text, and vision)
- Pinecone account and API key
- Windows, macOS, or Linux (tested primarily with Python/Streamlit)

Optional (improves ingestion of some formats):
- Poppler (for robust PDF processing); not required if sticking to PyPDFLoader
- Tesseract OCR (only if you plan to OCR images/PDFs outside the provided loaders)

---

## Quick Start

1) Clone and install

```bash
pip install -r requirements.txt
```

2) Configure environment

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_INDEX_NAME=concept-drift-explainer
```

Notes:
- The ingestion script will create the Pinecone index if it does not exist (serverless, `aws`/`us-east-1`).
- No region var is required for connecting; creation uses a default serverless spec.

3) Prepare documents and logs

- Put context documents in: `frontend/static/documents/`
  - Filenames should start with `YYYY-MM-DD_...` so the system can parse timestamps (e.g., `2017-10-11_Conference_Catering_Guidelines.pdf`).
- Ensure event logs are present in: `data/event_logs/<LogName>/` with three files:
  - `prediction_results.csv`
  - `window_info.json`
  - `<LogName>.xes`

4) Ingest documents and glossary

```bash
python -m backend.utils.ingest_documents
```

This will:
- Create/connect to the Pinecone index named in `PINECONE_INDEX_NAME`.
- Upsert document chunks into `context` namespace.
- Upsert BPM glossary terms into `bpm-kb` namespace.

5) Run the UI

```bash
streamlit run frontend/app.py
```

Then in your browser:
- Choose an event log under “Select Event Log to Analyze”.
- Click “Run Drift Analysis” to generate explanations.
- Inspect causes, download the DOCX report, and optionally ask follow‑up questions via the chatbot.

---

## Configuration & Environment

Environment variables (via `.env`):
- `OPENAI_API_KEY`: Required. Used by embeddings and GPT‑4o/4o‑mini.
- `PINECONE_API_KEY`: Required. Used by ingestion and retrieval.
- `PINECONE_INDEX_NAME`: Required. Name of the Pinecone index (created automatically if missing).

Models used:
- Text embeddings: `text-embedding-3-small`
- Text generation: `gpt-4o-mini`
- Vision (image analysis during ingestion): `gpt-4o`

---

## Usage Tips

- Document dating: The UI builds timelines and temporal filters using the date prefix in filenames. Prefer `YYYY-MM-DD_...` naming.
- Glossary: Keep `data/knowledge_base/bpm_glossary.csv` present to enrich reasoning; it is queried in a separate `bpm-kb` namespace and not cited as evidence.
- Caching: LLM responses are cached in `data/cache/llm_cache.json` to reduce cost and latency across runs.

---

## Testing & Evaluation

Evaluation assets and examples are under `tests/`. Useful entries include:
- `tests/run_master_evaluation.py`: Runs the pipeline across all logs in `data/event_logs/` and writes CSV reports.
- `tests/test_*`: Unit and scenario tests for core components.

You can also run scripts in `scripts/` to exercise parts of the pipeline for debugging.

---

## Troubleshooting

- “OPENAI_API_KEY not found…”: Ensure `.env` exists and the key is valid; restart the app/terminal after changes.
- Pinecone index errors: Verify `PINECONE_API_KEY` and `PINECONE_INDEX_NAME`; re‑run ingestion to create the index.
- No evidence found: Check that documents are ingested and date‑prefixed; verify the selected event log has valid `*.csv/json/xes` trio.
- Windows PDF ingestion: The pipeline uses `PyPDFLoader`; if switching to other loaders, you may need Poppler or additional dependencies.

---

## License

This is academic prototype code for a Master’s thesis. Unless otherwise specified, all rights reserved by the author. Contact the author for reuse permissions.

