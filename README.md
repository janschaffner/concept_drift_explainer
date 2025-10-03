# Concept Drift Explainer

A research prototype for explaining concept drift in business processes. The application links detected drift periods from process event logs with relevant internal and external context (policies, memos, slides, etc.) to produce concise, evidence-backed narratives about what changed and why. The system combines a multi-agent reasoning pipeline built with [LangGraph](https://github.com/langchain-ai/langgraph), a Pinecone vector database, and a Streamlit front end for interactive analysis.

---

## System Architecture

1. **Ingestion & Indexing**
   - Context documents placed in `frontend/static/documents/` are chunked, embedded with `text-embedding-3-small`, and upserted into the Pinecone `context` namespace. Filenames must begin with a `YYYY-MM-DD_` date prefix so timestamps can be inferred.
   - The BPM glossary at `data/knowledge_base/bpm_glossary.csv` is embedded into the `bpm-kb` namespace for terminology grounding.
2. **Explanation Pipeline (LangGraph)**
   - `drift_agent` derives a drift phrase, keywords, and case statistics from the selected event log window.
   - `context_retrieval_agent` performs semantic + temporal retrieval from Pinecone and constructs candidate evidence sets.
   - `re_ranker_agent` scores and curates the evidence, keeping a safety-net fallback snippet.
   - `franzoi_mapper_agent` maps snippets to the Franzoi context taxonomy, enriching the explanation structure.
   - `explanation_agent` synthesizes the final narrative with citations and confidence estimates; it also persists conversation state.
   - `chatbot_agent` enables iterative questioning using the accumulated graph state.
3. **Streamlit Interface**
   - The UI exposes workflows for managing documents, configuring settings, selecting drift windows, running the analysis, and exporting reports.

---

## Repository Layout

```
frontend/                # Streamlit application
  app.py                 # Main entry point
  pages/                 # Multi-page UI definitions (Home, Manage Context, Settings)
  static/documents/      # Source documents to ingest (user-provided)
backend/
  agents/                # LangGraph agent implementations
  graph/build_graph.py   # Assembles and compiles the workflow
  state/schema.py        # Shared GraphState contract
  utils/                 # Document ingestion, caching, embeddings, reporting helpers
data/
  event_logs/            # Event logs and detector outputs (CSV, JSON, XES)
  knowledge_base/        # BPM glossary CSV
  cache/                 # LLM response cache and temp files
scripts/                 # Convenience scripts for exercising the pipeline
tests/                   # Unit and evaluation scripts (master evaluation harness)
requirements.txt         # Python dependencies
```

---

## Prerequisites

- Python 3.10 or 3.11 (tested with CPython)
- [OpenAI API key](https://platform.openai.com/) with access to GPT-4o / GPT-4o-mini and `text-embedding-3-small`
- [Pinecone](https://www.pinecone.io/) account and API key (serverless index recommended)
- Optional: Poppler utilities (improves PDF parsing) and Tesseract OCR (only required if adding external OCR steps)

---

## Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-org>/concept_drift_explainer.git
cd concept_drift_explainer
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file at the repository root:

```env
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_INDEX_NAME=concept-drift-explainer
```

Notes:
- The ingestion script will create the Pinecone index on first run using the serverless `aws` / `us-east-1` defaults. Adjust the name or region if you prefer a different setup.
- No other Pinecone settings are required at runtime; the application reuses the index configured above.

### 5. Prepare data assets

1. **Context documents** – Place PDFs, PPTX, DOCX/TXT, PNG, or JPG files in `frontend/static/documents/`. Filenames must start with a date formatted as `YYYY-MM-DD_` (e.g., `2017-10-11_Conference_Guidelines.pdf`). Files without a parsable date are skipped.
2. **Event logs** – Each log folder under `data/event_logs/<LogName>/` should contain:
   - `prediction_results.csv` – Drift detector output
   - `window_info.json` – Metadata describing the detected drift windows
   - `<LogName>.xes` – Original XES process log
   Sample evaluation logs are provided in the repository.

---

## Using the Concept Drift Explainer

### Step 1 – Ingest documents and glossary

The ingestion step extracts text, generates embeddings, and upserts the vectors into Pinecone.

```bash
python -m backend.utils.ingest_documents
```

### Step 2 – Launch the Streamlit interface

```bash
streamlit run frontend/app.py
```

Open the URL printed in the terminal (defaults to http://localhost:8501). From the UI you can:
1. Select an event log under **“Select Event Log to Analyze.”**
2. Click **“Run Drift Analysis”** to execute the LangGraph pipeline.
3. Review the generated explanation, inspect individual evidence items, and download the DOCX report.
4. Use the built-in chatbot to ask follow-up questions grounded in the same evidence state.
5. Manage context documents or adjust configuration options from the other pages

### Step 3 – Export and iterate

- Use the **Download Report** button to export a DOCX summary of the explanation.
- Upload additional documents or refresh ingestion when new evidence becomes available.
- Clear cached LLM responses by deleting `data/cache/llm_cache.json` if you need to regenerate outputs from scratch.

---

## Command-Line Utilities & Scripts

- `scripts/run_full_chain_test.py` – Executes the full agent chain outside the UI (useful for debugging headless environments).
- `scripts/run_compiled_graph_test.py` – Compiles the LangGraph workflow to verify the state graph.
- `scripts/run_drift_agent_test.py` / `run_agent_chain_test.py` – Exercise specific agents.
- `backend/utils/clear_namespace.py` – Remove vectors from a Pinecone namespace.
Each script accepts `--help` for argument details (if applicable).
Evaluation assets are located in the `tests/` directory:
- `tests/run_master_evaluation.py` runs the pipeline across all event logs and aggregates metrics into CSV reports.
- `tests/test_drift_agent.py`, `test_glossary_visibility.py`, etc., cover unit-level behaviour.

Run the full evaluation harness:

```bash
python tests/run_master_evaluation.py
```

---

## Troubleshooting

| Symptom | Resolution |
| --- | --- |
| `ValueError: Pinecone API key or index name not found` | Confirm `.env` is present and variables are spelled correctly. Restart the terminal after editing. |
| No documents ingested | Ensure filenames start with `YYYY-MM-DD_` and the files are supported. Check the terminal for loader errors. |
| Streamlit app cannot connect to Pinecone | Verify the index exists in the configured region and that your API key has permissions. Rerun the ingestion script to recreate the index if necessary. |
| Repeated LLM calls are slow or expensive | Keep `data/cache/llm_cache.json`; deleting it forces regeneration. |



## TODO: CHANGE

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