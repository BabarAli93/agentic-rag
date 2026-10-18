import os
from langchain_core.tools import tool
from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from flashrank import Ranker, RerankRequest
from contextlib import asynccontextmanager

from retrieval.retriever import get_retriever
from services.source_manager import list_sources

model = init_chat_model(
    model="gpt-4o-mini"
)

#checkpointer = MemorySaver()
_db_checkpointer = None

# Path to persistent checkpointer SQLite file in storage/
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "storage", "checkpoints.sqlite")

# Initialize lightweight local cross-encoder reranker
ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")

SYSTEM_PROMPT = """
You are an advanced NotebookLM-style knowledge assistant. 
The user's documents and websites are stored in a hybrid vector database, and you MUST access them via the search_knowledge_base tool.

Important Rules:
- ALWAYS call search_knowledge_base when the user asks a question about their files, data, or websites.
- **Strict Citation Format:** Every factual claim MUST end with a source tag matching the exact source_name or source_id from search metadata.
- **FORMAT REQUIREMENT:** Output citations strictly as ``. 
  - Example PDF: `According to the resume...`
  - Example Website: `Based on the documentation...`
- NEVER output prefix words like "CITATION REF ->" or empty placeholders like "(e.g., )".
- If no information is found in the sources, explicitly state that it is not available.
"""


@asynccontextmanager
async def get_checkpointer_cm():
    """Async context manager for SQLite checkpointer to ensure thread safety per request."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(DB_PATH) as checkpointer:
        yield checkpointer


async def build_agent(owner_id: str, checkpointer=None):
    """Create an agent scoped to an owner with hybrid search, flashrank reranking, source citations, and persistent memory."""

    retriever = get_retriever(owner_id=owner_id)

    @tool
    def search_knowledge_base(query: str) -> str:
        """Search this user's indexed documents and websites using Hybrid Search and Neural Reranking.

        Args:
            query: The question, keywords, or topic to search for.

        Returns:
            The most relevant reranked document passages with source metadata and explicit citations.
        """

        # 1. Retrieve broad candidate pool (top 15) using Hybrid Search (Dense + Sparse)
        documents = retriever.invoke(query)

        if not documents:
            return "No relevant documents or web sources were found."

        # 2. Prepare documents for FlashRank neural reranking
        passages_to_rank = []
        for idx, doc in enumerate(documents, start=1):

            metadata = doc.metadata or {} 
            source_name = metadata.get("source_name") or metadata.get("source") or "Unknown Source"
            source_type = metadata.get("source_type", "document")

            passages_to_rank.append({
                "id": idx,
                "text": doc.page_content,
                "metadata": {
                    "source_name": source_name,
                    "source_type": source_type
                }
            })

        
        # 3. Execute FlashRank cross-encoder reranking
        rerank_request = RerankRequest(query=query, passages=passages_to_rank)
        ranked_results = ranker.rerank(rerank_request)

        # 4. Format top 4 highest-scoring reranked passages for the LLM agent
        results = []
        for index, item in enumerate(ranked_results[:4], start=1):
            meta = item.get("metadata", {})
            source_name = meta.get("source_name", "Unknown")
            source_type = meta.get("source_type", "document")
            content = item.get("text", "").strip()
            relevance_score = item.get("score", 0.0)

            results.append(
                f"""
[CITATION REF -> Source: {source_name}]
--- Reranked Result {index} (Score: {relevance_score:.4f}) ---
Source Name: {source_name}
Source Type: {source_type}

{content}
""".strip()
            )

        return "\n\n".join(results)

    @tool
    def list_indexed_sources() -> str:
        """List all documents and websites indexed for this user."""
        sources = list_sources(owner_id=owner_id)  # adjust to your signature
        if not sources:
            return "No sources have been indexed."
        return "\n".join(f"- {s['source_name']} ({s['source_type']}, {s['chunk_count']} chunks)" for s in sources)

    return create_deep_agent(
        model=model,
        tools=[search_knowledge_base, list_indexed_sources],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )