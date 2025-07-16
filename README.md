# Masterthesis

The focus of this Master's thesis is to bridge the gap between the detection of concept drifts in event logs and their practical interpretation. My initial research question is:  
**"How can external and internal context dimensions be integrated into process mining sense-making?"**

Currently, the CV4CDD framework for concept drift detection outputs change points in the form of JSON files. However, these are often not meaningful enough for analysts, as they lack explanations of *what* actually changed and *why*. 

My approach is to enrich these change points with relevant contextual data—such as organizational charts or new regulations—to help make sense of why a process has changed at a particular point in time. To achieve this, I use a large language model (specifically GPT-4o), which links the JSON output and event logs with contextual company data. This includes internal documents like Word files, PowerPoint presentations, intranet articles, and knowledge bases such as Confluence.

For example, if the intranet announces a new CIO taking office, the language model can associate that information with a nearby change point in the event logs. This helps analysts understand that a leadership change might be the underlying cause of the concept drift. In essence, the LLM serves as a bridge between the raw change points from the framework and the company-specific context, providing well-founded explanations for observed process changes.


# Entwurf ReadMe.md

# Concept Drift Explanation Engine
This repository contains the prototype for a Master's thesis project designed to bridge the gap between the technical detection of concept drifts in business process event logs and their practical, human-understandable interpretation.

The core research question is: "How can external and internal context dimensions be integrated into process mining sense-making?"

This system addresses the challenge that while process mining frameworks can detect when a process has changed, they don't explain why. This project uses a sophisticated pipeline of AI agents to analyze a detected drift, retrieve relevant contextual documents (e.g., internal memos, new regulations, presentation slides), and generate a clear, evidence-based explanation for the change.

# Core Features
Automated Explanation Generation: Ingests raw drift data and produces a structured JSON explanation with a summary and ranked, evidence-backed causes.

Multimodal Document Ingestion: Builds a comprehensive knowledge base by processing various document types, including PDFs, PowerPoint (.pptx) slides, Word (.docx) files, and even images (.png, .jpg) found within them.

Advanced AI Agent Pipeline: Utilizes LangGraph to orchestrate a multi-step agentic workflow where each agent has a specialized task, from retrieval and re-ranking to classification and synthesis.

Dual-Namespace Vector Search: Leverages a Pinecone vector database with separate namespaces for general context documents and a specialized BPM glossary, enabling hybrid search strategies.

Interactive Web Interface: A Streamlit application provides a user-friendly interface to run analyses, review explanations, upload new documents, and ask follow-up questions via an integrated chatbot.

Robust Evaluation Suite: Includes automated evaluation harnesses to rigorously test the pipeline's accuracy using metrics like Recall@2 and MRR.

# System Architecture
The system is built around a central, stateful graph of AI agents that process information sequentially.

1. Data Ingestion (ingest_documents.py)
Before analysis can begin, a knowledge base is created by the ingestion script. This script:

Scans a directory for all source documents (.pdf, .pptx, etc.) and a glossary file (.csv).

Extracts text content. For multimodal files like .pptx, it also extracts images and uses the GPT-4o Vision model to generate text descriptions.

Chunks the extracted text into manageable pieces.

Creates vector embeddings for each chunk using an OpenAI embedding model.

Upserts the vectors into a Pinecone index, attaching critical metadata like the source filename and a timestamp parsed from the name.

General documents are added to the context namespace.

Glossary terms are added to the bpm-kb namespace.

2. The Explanation Pipeline (build_graph.py)
When an analysis is triggered, the langgraph application executes a chain of agents:

Drift Agent: Parses the initial drift data (changepoints, timestamps, drift type) and extracts relevant keywords and specific entities (e.g., form IDs, project numbers) from the corresponding .xes event log to guide the search.

Context Retrieval Agent: Formulates a rich semantic query using the drift info and keywords. It queries both Pinecone namespaces:

It retrieves the top 30 candidate documents from the context namespace using a skewed temporal filter (prioritizing documents published 14 days before and 3 days after the drift).

It retrieves the single best-matching term from the bpm-kb glossary namespace.

Re-Ranker Agent: Intelligently filters the 30+ candidates down to a handful of highly relevant snippets.

It applies a heuristic date bonus to scores.

It force-keeps the single best candidate as a safety net.

It uses GPT-4o to re-rank the remaining candidates based on engineered features like a Specificity Score (how many unique entities a snippet contains).

Franzoi Mapper Agent: Classifies each of the final evidence snippets against the three-level Franzoi et al. (2025) context taxonomy (e.g., ORGANIZATION_INTERNAL::Process_Management), adding a layer of academic rigor.

Explanation Agent: Synthesizes the final output.

It dynamically selects a specialized prompt based on the drift type (Sudden, Gradual, Incremental, or Recurring).

It uses a two-step "draft and refine" chain to generate a high-quality, structured JSON explanation.

It performs a final confidence score calibration based on business rules (e.g., penalizing very old evidence).

Chatbot Agent: If the user asks a follow-up question in the UI, this agent is triggered. It uses the full context of the analysis and conversation history to provide an answer.

# Key Technologies
Backend: Python 3.10+

AI Orchestration: LangChain, LangGraph

LLMs: OpenAI (GPT-4o, GPT-4o-mini)

Vector Database: Pinecone

Web UI: Streamlit

Data Handling: Pandas

# Getting Started
Follow these steps to set up and run the project locally.

1. Prerequisites
Python 3.10 or higher

An OpenAI API Key

A Pinecone API Key

2. Installation
Clone the repository and install the required dependencies. It is recommended to use a virtual environment.

Bash

git clone https://github.com/your-repo/masterthesis.git
cd masterthesis
pip install -r requirements.txt
3. Configuration
Create a .env file in the project's root directory. This file stores your secret API keys.

Code-Snippet

OPENAI_API_KEY="sk-..."
PINECONE_API_KEY="your-pinecone-api-key"
PINECONE_INDEX_NAME="concept-drift-explainer"
4. Data Ingestion
Before running the application for the first time, you must populate your Pinecone vector database. Place your source documents in the data/documents/ folder and your glossary in data/knowledge_base/. Then, run the ingestion script:

Bash

python -m backend.utils.ingest_documents
This will create the Pinecone index if it doesn't exist and upload all processed documents.

5. Running the Application
Launch the Streamlit web interface:

Bash

streamlit run ui/app.py
Open your web browser to the local URL provided by Streamlit to start using the application.

# Testing and Evaluation
The project includes two evaluation scripts located in the /tests directory.

run_single_evaluation.py: A lightweight script for quickly testing the pipeline against the single test case in data/drift_outputs/.

run_master_evaluation.py: A comprehensive harness that runs the pipeline against all test sets found in data/event_logs/, generating a final master_eval_report.csv with performance metrics.