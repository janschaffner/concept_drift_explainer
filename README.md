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



The heart of the application's backend is the orchestration and state management system, which governs the agentic workflow.
The primary orchestrator is located in \textit{backend/graph/build\_graph.py}, which is responsible for assembling and compiling the complete LangGraph application.
This script registers each agent as a distinct node.
It also defines the linear sequence of the main analytical pipeline through directed edges, and implements the conditional routing logic via the \textit{should\_continue} function. 
Additionally, this script centralizes the initialization of shared resources, such as the Pinecone database connection, which is then partially applied to the agents that require it.
The active workflow is complemented by the passive data contract, which is defined in \textit{backend/state/schema.py}.
This file serves as the application's information model's single source of truth and formally specifies the \textit{GraphState}, as discussed in Section \ref{sec:4.3}.

The agents and utilities are modularized within the backend.
The systems' core logic, embodied by the agents, is encapsulated within self-contained modules.
These modules are located in the designated \textit{backend/agents/} directory.
Each file corresponds to a specific agent in the workflow, from the initial data ingestion and abstraction in \textit{drift\_agent.py}, through the semantic retrieval and re-ranking in \textit{context\_retrieval\_agent.py} and \textit{re\_ranker\_ agent.py}, to the final synthesis and user interaction in \textit{explanation\_agent.py} and \textit{chatbot\_agent.py}.
The standalone \textit{drift\_linker\_agent.py}, which contains the logic for the drift meta-analysis, is also included here.
Utilities and shared helper functions are organized in the \textit{backend/utils/} directory.
They include modules for managing LLM response caching (\textit{cache.py}), generating vector embeddings for the \textit{drift\_phrase} (\textit{embeddings.py}) and exporting final reports from the frontend (\textit{reporting.py}).
A dedicated script for analyzing and describing images in general or in PowerPoint presentations (\textit{image\_analyzer.py}) is also located here, as well as the handling of the data ingestion process for new context documents into the vector database (\textit{ingest\_documents.py}).
Finally, \textit{clear\_namespace.py} has the capacity to reset a namespace in the vector database.

The remaining directories contain the data fixtures, the user interface, and the evaluation scripts.
The \textit{data/} directory contains all persistent data, knowledge bases, and evaluation assets utilized by the application.
This includes the curated BPM glossary (\textit{data/knowledge\_base/}), the raw event logs and detector outputs (\textit{data/event\_logs/}, \textit{data/drift\_outputs/}), the unstructured documents for the retrieval corpus (\textit{data/testset/context\_ documents/}), and the persistent caches and feedback logs (\textit{data/cache/llm\_cache.json}, \textit{data/feedback/feedback\_log.jsonl}).
The user interface is implemented as a Streamlit application, with the main entry point located in \textit{frontend/app.py}
The application is structured using Streamlit's multi-page app format, with distinct pages for the main dashboard, context management, and settings.
The pages are defined within the \textit{frontend/pages/} directory.
All static assets, including images and documents for direct access, are stored in \textit{frontend/static/} and \textit{frontend/assets/}.
Finally, the repository contains directories designated for testing and evaluation purposes. 
The \textit{tests/} directory contains the master evaluation harness, which is utilized to calculate performance metrics such as Recall@k and the Mean Reciprocal Rank (MRR). 
The \textit{scripts/} directory offers executable entry points for diverse component and end-to-end evaluations.