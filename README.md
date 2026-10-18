# Agentic RAG

Agentic RAG is a document and website question-answering workbench. Add PDF or TXT files and web pages as sources, inspect their extracted chunks, and ask questions in a streaming chat. The agent retrieves source-grounded passages, reranks them, and uses them to answer with source references.

## Features

- **Multiple source types:** upload PDF and TXT files, or index a web page from its HTTP(S) URL.
- **Source management:** view indexed sources and chunk counts, inspect extracted chunks, and remove a source and its vectors.
- **Streaming chat:** receive generated responses as they are produced, with chat history persisted per thread.
- **Grounded answers:** the agent searches the knowledge base for questions about sources and formats source references as interactive citations in the UI.
- **Hybrid retrieval and reranking:** combine dense and sparse Qdrant search, retrieve an initial pool of up to 15 passages, and rerank candidates locally with FlashRank before supplying up to four passages to the agent.
- **Per-browser workspaces:** the UI keeps a generated owner ID in browser local storage and a separate thread ID per tab.

## Technology

- Python 3.13+
- FastAPI and Uvicorn
- LangChain and Deep Agents
- Qdrant for vector storage and dense/sparse hybrid retrieval
- OpenAI `gpt-4o-mini` for chat and `text-embedding-3-small` for embeddings
- FlashRank `ms-marco-MiniLM-L-12-v2` for local passage reranking
- SQLite for the source registry and LangGraph chat checkpoints
- `uv` for Python dependency management

## Requirements

- Python 3.13 or later
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Docker with the Docker Compose plugin
- An OpenAI API key
- Optional: a LangSmith API key and tracing settings

## Setup

1. Install the project dependencies:

   ```bash
   uv sync
   ```

2. Create a local environment file:

   ```bash
   cp .env.example .env
   ```

3. Set `OPENAI_API_KEY` in `.env`. LangSmith settings are optional.

   ```dotenv
   OPENAI_API_KEY=your_openai_api_key_here
   LANGSMITH_API_KEY=your_langsmith_api_key_here

   QDRANT_URL=http://localhost:6333
   QDRANT_COLLECTION=agentic_rag

   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_PROJECT=agentic-rag
   ```

   `OPENAI_API_KEY` is required for chat and embeddings. Do not commit `.env` or share API keys.

## Start the services

Start Qdrant:

```bash
docker compose up -d
```

The Compose setup exposes Qdrant on ports `6333` (HTTP/API) and `6334` (gRPC), and keeps its data in the `qdrant_storage` Docker volume. The Qdrant dashboard is at [http://localhost:6333/dashboard](http://localhost:6333/dashboard).

Start the web app from the project directory:

```bash
uv run app.py
```

Open [http://localhost:8000](http://localhost:8000). FastAPI's interactive API documentation is available at [http://localhost:8000/docs](http://localhost:8000/docs).

Stop Qdrant when finished:

```bash
docker compose down
```

## Use the workbench

1. Add a website using its HTTP(S) URL, or upload one or more PDF or TXT files in the left-hand source panel.
2. Wait for the indexing confirmation. The source list shows each source's type and indexed chunk count.
3. Select a source to inspect its extracted chunks. Use its delete control to remove it from the registry and Qdrant.
4. Ask a question in the chat. Answers stream into the conversation; click a citation to inspect the referenced source.
5. Use **New Chat** to start a separate conversation thread. Existing thread history is restored when reopening that tab.

Documents are split into chunks of 1,000 characters with 200 characters of overlap. Qdrant performs hybrid dense and sparse retrieval over up to 15 candidates, and FlashRank reranks them to select up to four passages for the agent.

## API overview

The FastAPI application exposes these routes:

| Method | Route | Purpose |
| --- | --- | --- |
| `GET` | `/api/sources` | List sources for an owner |
| `POST` | `/api/sources/file` | Upload and index a PDF or TXT file |
| `POST` | `/api/sources/url` | Fetch and index a web page |
| `DELETE` | `/api/sources/{source_id}` | Remove a source and its vectors |
| `GET` | `/api/sources/{source_id}/chunks` | Retrieve a source's indexed chunks |
| `POST` | `/api/chat/stream` | Stream chat output using server-sent events |
| `POST` | `/api/chat` | Return a non-streaming chat response |
| `GET` | `/api/chat/history` | Read persisted history for a thread |

The interactive request/response schemas are documented at `/docs`.

## Configuration

| Variable | Purpose | Default |
| --- | --- | --- |
| `OPENAI_API_KEY` | Required for the chat model and embeddings | None |
| `QDRANT_URL` | Qdrant server URL | `http://localhost:6333` |
| `QDRANT_COLLECTION` | Qdrant collection used for indexed chunks | `agentic_rag` |
| `LANGSMITH_API_KEY` | Optional LangSmith API key | None |
| `LANGCHAIN_TRACING_V2` | Enables LangChain tracing when configured | Not set |
| `LANGCHAIN_PROJECT` | LangSmith project name | Not set |

## Project structure

```text
agents/       Agent tools, prompting, reranking, and SQLite checkpoint setup
ingestion/    PDF/TXT and web page loaders, chunking, and indexing
models/       OpenAI embedding configuration
retrieval/    Owner-filtered Qdrant retrieval
services/     Source lifecycle and ingestion orchestration
static/       Browser-based source workbench and chat UI
storage/      Qdrant setup and SQLite source registry
app.py        FastAPI application and API routes
config.py     Environment-variable configuration
docker-compose.yml  Local Qdrant service
```

## Data and privacy notes

- Qdrant vectors persist in the Docker volume. The SQLite source registry and chat checkpoint database are stored under `storage/`.
- The browser creates an owner ID locally to keep its sources separate from other browser workspaces. This is a convenience mechanism, **not authentication or a security boundary**. The current app has no user login; do not expose it to untrusted users or the public internet without adding authentication, authorization, and deployment hardening.
- OpenAI receives chat prompts and text used to generate embeddings. Review the providers' data policies before indexing sensitive content.
