# Agentic RAG System

A locally deployed agentic RAG system for document ingestion, semantic search, and LLM-based question answering.
It used Inngest, OpenAI, Docker, Qdrant DB, and Node.js backend to run this applicaiton.

## Features

- PDF ingestion
- Document chunking
- Embedding generation
- Qdrant vector search
- Agentic retrieval
- Local LLM inference
- Inngest workflow orchestration

## Architecture

PDF → Chunking → Embeddings → Qdrant
                                      ↓
User Query → Agent → Retrieval → LLM → Response

## Tech Stack

- Python
- uv
- Qdrant
- Inngest
- OpenAI 

## Prerequisites

Tools required:
- Docker (Docker Desktop)
- Node.js
- OpenAI API Key 


## Installation and Application Running

Clone the code in your system
```bash
git clone repo_name
cd repo 
```

### Node.js Backend

Run this command from a terminal inside the project directory
```bash
npx inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest --no-discovery
```

### Qdrant DB Running
Run this command in a separate terminal inside the working directory for Qdrant DB
```bash
docker run -d --name qdrantRagDb -p 6333:6333 -v "./qdrant_storage:/qdrant/storage" qdrant/qdrant
```

### Main application

After the live Qdrant DB and the Node.js backend, run this command to live this application
```bash 
uv run uvicorn main:app
```

Open the link 'http://localhost:8288' in any browser to navigate through the Inngest dashboard


### Frontend 
In a seprate terminal, run this command:
```bash 
uv run streamlit run streamlit_app.py
```

Now open the 'http://localhost:8501' link in any browser to use this application