# Concept Drift Explainer

This repository contains the source code for the Concept Drift Explainer (CDE), a tool developed 

## Abstract

Concept drift detection in process mining identifies process changes but often fails to explain their underlying causes, creating a detection-interpretation gap. 
This thesis addresses this problem by designing and evaluating the Concept Drift Explainer (CDE), a prototypical artifact that provides context-aware explanations for such drifts. Using a Design Science Research methodology, an LLM-based multi-agent system was developed that systematically links drift signals from event logs to evidence from unstructured enterprise knowledge sources. The artifact was evaluated through a controlled experiment using real-world event logs and a synthetic context corpus. 
It achieved high accuracy in identifying the correct explanatory documents. Subsequent expert evaluation confirmed the artifact's practical utility, demonstrating its ability to significantly enhance analytical efficiency by generating traceable, evidence-based hypotheses. This thesis contributes to the research by operationalizing context-aware sense-making in process mining. It also offers a design pattern for trustworthy, AI-driven, explanatory tools for practitioners.

---

## Use the CDE as a WebApp

The Concept Drift Explainer is deployed via Streamlit and accessible at https://concept-drift-explainer.streamlit.app/.
By default, the web app is set to private. To access it, please create a [Streamlit Account](https://share.streamlit.io/) and provide your email address used to the author or request public access for a limited time for a day or two (<mailto:jschaffn@uni-muenster.de> or DM via Slack). This way, you will not need to clone and set up the entire application. Streamlit mirrors the entire GitHub repository.

---

## System Architecture

1. **Ingestion & Indexing**
   - Context documents placed in `frontend/static/documents/` are chunked, embedded with `text-embedding-3-small`, and upserted into the Pinecone `context` namespace. Filenames must begin with a `YYYY-MM-DD_` date prefix so timestamps can be inferred.
   - The BPM glossary at `data/knowledge_base/bpm_glossary.csv` is embedded into the `bpm-kb` namespace for terminology grounding.
2. **Explanation Pipeline (LangGraph)**
   - `Orchestrator` assembles and compiles the full agentic workflow, manages the shared state, and routes control flow between all other agents.
   - `drift_agent` derives a drift phrase, keywords, and case statistics from the selected event log window.
   - `context_retrieval_agent` performs semantic + temporal retrieval from Pinecone and constructs candidate evidence sets.
   - `re_ranker_agent` scores and curates the evidence, keeping a safety-net fallback snippet.
   - `franzoi_mapper_agent` maps snippets to the Franzoi context taxonomy, enriching the explanation structure.
   - `explanation_agent` synthesizes the final narrative with citations and confidence estimates; it also persists conversation state.
   - `chatbot_agent` enables iterative questioning using the accumulated graph state.
   - `drift_linker_agent` performs a meta-analysis after multiple runs to identify and summarize relationships between different drifts.
3. **Streamlit Interface**
   - The UI exposes workflows for managing documents, configuring settings, selecting drift windows, running the analysis, and exporting reports.

---

## Repository Layout

```
backend/                          # Orchestration, state management, and agentic workflow
  agents/                         # Self-contained agent modules
    drift_agent.py                # Data ingestion and abstraction agent
    context_retrieval_agent.py    # Semantic retrieval agent
    re_ranker_agent.py            # Re-ranking agent
    explanation_agent.py          # Final synthesis agent
    chatbot_agent.py              # User interaction agent
    drift_linker_agent.py         # Drift meta-analysis agent
  graph/build_graph.py            # Assembles and compiles the LangGraph application
  state/schema.py                 # Defines the shared GraphState data contract
  utils/                          # Shared helper functions and utilities
    cache.py                      # LLM response caching management
    embeddings.py                 # Vector embedding generation
    reporting.py                  # Report generation from the frontend
    image_analyzer.py             # Image and PowerPoint analysis
    ingest_documents.py           # Handles ingestion of new context documents
    clear_namespace.py            # Resets a namespace in the vector database
frontend/                         # Streamlit user interface
  app.py                          # Main application entry point --> run `streamlit run frontend/app.py`
  pages/                          # Multi-page UI definitions (Dashboard, Context Management, Settings)
  static/                         # Static assets for direct user access
    documents/                    # Primary source documents for ingestion by the application
  assets/                         # Other static assets (e.g., images)
data/                             # Persistent data, knowledge bases, and evaluation assets
  documents/                      # For local, standalone testing of utility scripts only
  knowledge_base/                 # Curated BPM glossary
  event_logs/                     # Primary input data for the master evaluation harness (/tests/run_master_evaluation.py)
  drift_outputs/                  # Default or fallback location for the raw drift detection data + single tests directory
  testset/                        # Sandbox for testing components
    context_documents/            # Master archive for the synthetically generated context document corpus used for evaluating the CDE
    cv4cdd_output/                # Master archive for all evaluation event logs used by the CDE
  cache/                          # Persistent caches
    llm_cache.json                # LLM response cache
  feedback/                       # Feedback logs
    feedback_log.jsonl            # User feedback log
scripts/                          # Executable entry points for various evaluations and tests during development
tests/                            # Master evaluation harness for performance metrics (Recall@k, MRR)
```

---

## Prerequisites

- Python 3.9 or newer
- [OpenAI API key](https://platform.openai.com/) with access to GPT-4o / GPT-4o-mini and `text-embedding-3-small`
- [Pinecone](https://www.pinecone.io/) account, API key and a Pinecone index (index name set to `conceptdriftexplainer`)

---

## Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/janschaffner/concept_drift_explainer.git
cd concept_drift_explainer
```

### 2. Create a virtual environment (recommended)

macOS / Linux:
```bash
python -m venv .venv
source .venv/bin/activate
```

Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file at the repository root:

```env
OPENAI_API_KEY=your-openai-api-key
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_INDEX_NAME=conceptdriftexplainer
```

Notes:
- The ingestion script will create the Pinecone index on first run using the serverless `aws` / `us-east-1` defaults. Adjust the name or region if you prefer a different setup.
- No other Pinecone settings are required at runtime; the application reuses the index configured above.

### 5. Prepare data assets

1. **Context documents** – Place PDFs, PPTX, DOCX/TXT, PNG, or JPG files in `frontend/static/documents/`. Filenames must start with a date formatted as `YYYY-MM-DD_` (e.g., `2017-10-11_Conference_Guidelines.pdf`). Files without a parsable date are skipped. The synthetic context document corpus used for evaluation is already stored there.
2. **Event logs** – Each log folder under `data/event_logs/<LogName>/` should contain:
   - `prediction_results.csv` – CV4CDD-4D drift detector output (see: [CV4CDD-4D](https://gitlab.uni-mannheim.de/processanalytics/cv4cdd.git/))
   - `window_info.json` – Metadata describing the detected drift windows (also given by CV4CDD-4D)
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
Evaluation assets are located in the `tests/` directory:
- `tests/run_master_evaluation.py` runs the pipeline across all event logs and aggregates metrics into CSV reports.

Run the full evaluation harness that was used for Eval3:

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