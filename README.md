# 🚀 Agentic RAG Workbench

**Release:** v1.3.0

A tenant-aware, enterprise-grade document, media, and website question-answering workbench built with **FastAPI**, **LangGraph**, **Keycloak (OIDC/JWT)**, **Qdrant**, and **Tailwind CSS**. The platform orchestrates stateful multi-node agents that route queries, perform tenant-isolated hybrid retrieval with local reranking, and run corrective grading loops to guarantee grounded, cited responses across diverse LLM providers.

## ✨ Key features

- **Enterprise Security & Keycloak OIDC/JWT (v1.3.0):**
  - Verify Keycloak-compatible JWTs, enforce role-based access, scope sources by authenticated tenant, and scope chat history by tenant and user.
  - Production-grade RS256 token verification via `PyJWKClient` and Keycloak identity provider integration.
  - Strict JWT validation enforcing core standard claims (`exp`, `iat`, `iss`, `sub`, `aud`) and structural safeguards.
  - User attribute mapping for multi-tenant data isolation (`tenant_id`), eliminating untrusted client-side identities.
  - Integrated OIDC login/logout controls, auto-refreshing token loops, and dynamic user/tenant badge rendering in the UI.

- **Fail-Closed Input & Output Guardrails:**
  - **Input Guardrails:** Multi-stage prompt injection detection using pattern matching and fail-closed LLM safety classification.
  - **Output Guardrails:** Automated PII and credential leak detection (SSN, API keys, private tokens, credit cards) before sending responses to clients.

- **Expanded Ingestion & Source Management:**
  - **Multimodal Support:** Upload PDF, TXT, Markdown, and DOCX files; index web pages and YouTube transcripts.
  - **Full Lifecycle Management:** Inspect chunked text excerpts, delete sources from storage, and re-index existing sources on demand.
  - **Smart Re-indexing:** Compares content hashes to skip re-embedding when source content has not changed, conserving API quota and compute.
  - **Optimized Chunking:** Context-preserving splitters (2,000 characters with 400 overlap) balancing semantic context preservation across chunk boundaries.

- **Multi-Provider Model Ecosystem:**
  - Flexible provider selection across **OpenAI** (`gpt-4o-mini`), **HuggingFace**, **Ollama** (`qwen2.5`), and OpenAI-compatible **vLLM** endpoints via LangChain connectors.

- **Stateful LangGraph Agent & Corrective RAG:**
  - Explicit workflow nodes for query routing, query rewriting, hybrid retrieval, evidence grading, answer generation, and grounding checks.
  - Automated retry loops (up to 2 retries) with query refinement before falling back to graceful abstention.
  - SQLite persistence via `langgraph-checkpoint-sqlite` to maintain thread state and citation evidence maps across sessions.

- **Hybrid Search & Reranking:**
  - Dense semantic + sparse keyword search via **Qdrant**, strictly filtered by `tenant_id`.
  - Local **FlashRank** (`ms-marco-MiniLM-L-12-v2`) reranking to reduce candidate chunks to top-scoring passages.

- **Evaluation & Observability:**
  - Integrated RAG evaluation pipeline powered by **Ragas** for measuring faithfulness and answer relevance.
  - Optional **LangSmith** tracing for end-to-end agent execution inspection.

- **Interactive UI & Streaming:**
  - Real-time status indicators, token-by-token SSE response streaming, interactive evidence panel, and source chunk inspection modal.

## 🧩 Agent Architecture

```mermaid
flowchart TD
    START([User message]) --> Auth{JWT / OIDC Auth Check}
    Auth -->|Unauthorized| 401[Return 401 Bearer Challenge]
    Auth -->|Valid Tenant JWT| InGuard{Input Safety Guardrail}
    
    InGuard -->|Injection / Unsafe| Blocked[Block Request & Return Reason]
    InGuard -->|Safe Input| Router{Query router}

    Router -->|Greeting / non-factual| GeneralChat[General chat]
    GeneralChat --> OutGuard

    Router -->|Factual question| Rewriter[Rewrite search query]
    Rewriter --> Retrieval[Tenant-Isolated Hybrid Retrieval & Qdrant]
    Retrieval --> Rerank[FlashRank reranking]
    Rerank --> Grader{Evidence grader}

    Grader -->|Sufficient evidence| Generator[Generate source-grounded answer]
    Grader -->|Insufficient evidence| CheckRetry1{Retry count less than 2?}

    Generator --> Grounding{Grounding checker}
    Grounding -->|Grounded| OutGuard{Output PII Guardrail}
    Grounding -->|Not grounded| CheckRetry2{Retry count less than 2?}

    CheckRetry1 -->|Yes| Refine[Refine search query]
    CheckRetry1 -->|No| Abstain[Abstain / fallback]
    CheckRetry2 -->|Yes| Refine
    CheckRetry2 -->|No| Abstain

    Refine --> Retrieval
    Abstain --> OutGuard

    OutGuard -->|Safe Output| END2([Stream Response to Client])
    OutGuard -->|PII / Credential Leak| Sanitized[Redact / Block Response]
```

## Technology Stack

- **Backend:** Python 3.13+, FastAPI, Uvicorn, PyJWT (`PyJWKClient`)
- **Identity Provider:** Keycloak 24+ (OpenID Connect / OAuth2)
- **Agent Orchestration:** LangGraph, LangChain, SQLite Checkpointer
- **Reranking & Evaluation:** FlashRank (`ms-marco-MiniLM-L-12-v2`), Ragas
- **Vector Storage:** Qdrant (Dense + Sparse Hybrid Search)
- **Model Connectors:** LangChain OpenAI, HuggingFace, Ollama
- **Dependency & Package Manager:** [uv](https://docs.astral.sh/uv/)

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)
- Docker with Docker Compose plugin
- Keycloak Server 24+ (configured with realm `agentic-rag` and client `agentic-rag-backend`)
- OpenAI / Ollama API credentials

---

## Setup & Running

1. **Install dependencies:**
   ```bash
   uv sync
   ```

2. **Configure environment variables:**
   ```bash
   cp .env.example .env
   ```
   Edit `.env` to configure your Keycloak, Qdrant, and LLM provider parameters.

3. **Start Storage & Auth Services:**
   ```bash
   docker compose up -d
   ```
   *(Qdrant Dashboard at `http://localhost:6333/dashboard`; Keycloak Console at `http://localhost:8080/admin/`)*

4. **Launch Application:**
   ```bash
   uv run app.py
   ```
   Access the workbench UI at [http://localhost:8000](http://localhost:8000). Interactive API specs are available at `/docs`.

---

## Configuration Reference

| Variable | Description | Default |
| --- | --- | --- |
| `ENVIRONMENT` | Deployment stage (`development` or `production`) | `production` |
| `AUTH_MODE` | Authentication mode (`oidc` or `development`) | `oidc` |
| `OIDC_ISSUER` | Keycloak realm issuer URL | `http://localhost:8080/realms/agentic-rag` |
| `OIDC_AUDIENCE` | Expected JWT audience claim (`aud`) | `agentic-rag-api` |
| `OIDC_ROLE_CLIENT_ID` | Client ID for role extraction | `agentic-rag-backend` |
| `OIDC_TENANT_CLAIM` | JWT token claim mapping tenant ID | `tenant_id` |
| `DEFAULT_TENANT_ID` | Fallback tenant identifier | `agentic-rag` |
| `OPENAI_API_KEY` | OpenAI API key | None |
| `QDRANT_URL` | Qdrant server endpoint | `http://localhost:6333` |
| `QDRANT_COLLECTION` | Qdrant collection name | `agentic_rag` |
| `DEPLOYMENT_MODE` | Active LLM model provider (`remote`, `openai`, `local`) | `remote` |
| `VLLM_BASE_URL` | vLLM server URL (`remote` mode) | None |
| `VLLM_MODEL_NAME` | Model ID for `remote` mode | `Qwen/Qwen3-32B` |
| `DEV_AUTH_ENABLED` | Enable mock authentication in dev mode | `false` |

---

## API Overview

| Method | Route | Security | Purpose |
| --- | --- | --- | --- |
| `GET` | `/api/sources` | Bearer JWT | List tenant-isolated indexed sources |
| `POST` | `/api/sources/file` | Bearer JWT | Upload and index PDF, TXT, MD, or DOCX documents |
| `POST` | `/api/sources/url` | Bearer JWT | Fetch and index web pages or YouTube transcripts |
| `DELETE` | `/api/sources/{source_id}` | Bearer JWT | Remove a source and its vector payload records |
| `GET` | `/api/sources/{source_id}/chunks` | Bearer JWT | Inspect indexed text chunks for a source |
| `POST` | `/api/sources/{source_id}/reindex` | Bearer JWT | Re-parse and re-index an existing source (skips embedding if unchanged) |
| `POST` | `/api/chat/stream` | Bearer JWT | Stream agent progression and text generation via SSE |
| `POST` | `/api/chat` | Bearer JWT | Direct non-streaming chat endpoint |
| `GET` | `/api/chat/history` | Bearer JWT | Retrieve thread state history for authenticated tenant |

---

## Project Structure

```text
agents/       LangGraph workflow definitions, state models, and agent nodes
ingestion/    Parsers (PDF, DOCX, TXT, MD, Web, YouTube), splitters, and vector indexers
models/       Multi-provider model connectors (OpenAI, HuggingFace, Ollama)
retrieval/    Tenant-isolated hybrid search and FlashRank reranking engine
security/     Keycloak RS256 JWT validation, claims extraction, and fail-closed guardrails
services/     Pipeline orchestration, evaluation (Ragas), and source lifecycle
static/       OIDC-authenticated web application interface and streaming UI
storage/      Qdrant client initialization and SQLite state persistence
app.py        FastAPI application setup, CORS middleware, and API endpoints
config.py     Environment variable schemas and settings management
```

## 🔒 Security & Data Privacy
- **Multi-Tenant Isolation:** All vector search filters, document metadata, and chat checkpoints strictly enforce boundaries using the validated `tenant_id` extracted from the Keycloak JWT.
- **LLM Safety Guardrails:** Every user query passes through pattern and LLM-based prompt injection detection. Model outputs are scanned for PII and credential leaks before client delivery.
- **Provider Choice:** Supports completely air-gapped local setups (`Ollama` + local Qdrant) or cloud endpoints (`OpenAI`).