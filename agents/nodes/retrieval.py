from langchain_core.messages import HumanMessage
from flashrank import RerankRequest
from models.required_models import get_chat_model, get_ranking_model
from retrieval.retriever import get_retriever
from agents.state import AgentState

ranker = get_ranking_model()
model = get_chat_model()

async def rewriter_node(state: AgentState):
    history = state["messages"]
    last_user_msg = next((msg.content for msg in reversed(history) if msg.type in ("human", "user")), "")
    recent_history = history[-5:-1]

    rewrite_prompt = (
        "Write one standalone search query for retrieving evidence from the user's "
        "indexed sources. Use recent conversation context to resolve pronouns and carry forward "
        "the specific entity being discussed. Preserve the full scope of the question. "
        "Return ONLY the raw search query string with no quotes or explanations.\n"
        f"Recent conversation:\n"
        + "\n".join(f"{msg.type}: {msg.content}" for msg in recent_history)
        + f"\nLatest question: {last_user_msg}"
    )

    res = await model.ainvoke([HumanMessage(content=rewrite_prompt)])
    return {"search_query": res.content.strip().replace('"', '')}


async def retrieval_and_rerank_node(state: AgentState):
    owner_id = state.get("owner_id", "default_user")
    retriever = get_retriever(owner_id=owner_id)

    query = state.get("search_query") or state["messages"][-1].content
    documents = retriever.invoke(query)

    if not documents:
        # Clear out previous stale documents so grader doesn't re-evaluate old failed context
        return {"documents": "", "evidence_sufficient": False}

    passages = []
    for idx, doc in enumerate(documents, start=1):
        meta = doc.metadata or {}
        source_name = meta.get("source_name") or meta.get("source") or "Indexed Source"
        passages.append({
            "id": idx,
            "text": doc.page_content,
            "metadata": {
                "source_name": source_name,
                "chunk_index": meta.get("chunk_index", idx)
            }
        })

    # Rerank retrieved passages using local cross-encoder
    rerank_req = RerankRequest(query=query, passages=passages)
    ranked = ranker.rerank(rerank_req)

    # Clean, token-efficient passage formatting
    formatted_results = []
    for item in ranked[:8]:
        metadata = item.get("metadata") or {}
        s_name = metadata.get("source_name") or metadata.get("source") or "Indexed Source"
        content = item.get("text", "").strip()

        # Include clear, lightweight source tags for LLM citation placement
        formatted_block = f"\n{content}"
        if formatted_block not in formatted_results:
            formatted_results.append(formatted_block)

    final_documents = "\n\n---\n\n".join(formatted_results)
    return {"documents": final_documents}