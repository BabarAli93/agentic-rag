# Agentic RAG

A document question-answering app built with Chainlit, LangChain, and Qdrant. Upload PDF or TXT files in the chat, then ask questions grounded in their contents. The app chunks and embeds uploaded documents, stores them in Qdrant, and uses an OpenAI model to answer questions with retrieved context.

## Requirements

- Python 3.13 or later
- [uv](https://docs.astral.sh/uv/getting-started/installation/) for Python environment and dependency management
- Docker with Docker Compose
- An OpenAI API key
- Optional: a LangSmith API key for tracing

## Setup

1. From the project directory, install the dependencies:

	```bash
	uv sync
	```

2. Create a local environment file from the template:

	```bash
	cp .env.example .env
	```

3. Add your OpenAI key to `.env`. LangSmith settings are optional; add a LangSmith API key if you want tracing.

	```dotenv
	OPENAI_API_KEY=your_openai_api_key_here
	LANGSMITH_API_KEY=your_langsmith_api_key_here

	QDRANT_URL=http://localhost:6333
	QDRANT_COLLECTION=agentic_rag

	LANGCHAIN_TRACING_V2=true
	LANGCHAIN_PROJECT=agentic-rag
	```

	The OpenAI key is used for both chat responses and embeddings. If you do not use LangSmith, you can omit its key and tracing variables.

## Start Qdrant

Start the local Qdrant service using the included Compose file:

```bash
docker compose up -d
```

Qdrant's dashboard is available at [http://localhost:6333/dashboard](http://localhost:6333/dashboard). The Compose configuration stores Qdrant data in a named Docker volume, so indexed documents remain available after the container is restarted.

To stop Qdrant:

```bash
docker compose down
```

## Run the app

With Qdrant running and `.env` configured, start Chainlit:

```bash
uv run chainlit run app.py -w
```

Open the local URL printed in the terminal, typically [http://localhost:8000](http://localhost:8000).

## Use

1. Upload one or more PDF or TXT files in the chat. The app reports how many text chunks were indexed.
2. Ask a question in the chat. The agent searches the indexed documents and uses relevant passages to answer.

Uploaded documents are split into chunks of 1,000 characters with 200 characters of overlap. Retrieval returns up to five passages per search. Documents are stored in the configured Qdrant collection.

## Project structure

```text
agents/       Agent and knowledge-base search tool
ingestion/    PDF/TXT and web document loaders, chunking, and indexing
models/       OpenAI embedding configuration
retrieval/    Qdrant-backed document retrieval
storage/      Qdrant client and vector store setup
app.py        Chainlit chat application
config.py     Environment-variable configuration
docker-compose.yml  Local Qdrant service
```

The web page loader is available in `ingestion/loader.py`, but the current chat interface only accepts PDF and TXT uploads.

## Configuration

| Variable | Purpose | Default |
| --- | --- | --- |
| `OPENAI_API_KEY` | Required for the chat model and embeddings | None |
| `LANGSMITH_API_KEY` | Optional LangSmith tracing key | None |
| `QDRANT_URL` | Qdrant server URL | `http://localhost:6333` |
| `QDRANT_COLLECTION` | Collection used for indexed documents | `agentic_rag` |
| `LANGCHAIN_TRACING_V2` | Enables LangChain tracing when configured | Not set |
| `LANGCHAIN_PROJECT` | LangSmith project name | Not set |

Keep API keys in `.env`; do not commit that file. `.env.example` is provided as a safe template.