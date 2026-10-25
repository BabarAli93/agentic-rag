import logging

from langchain_core.messages import HumanMessage
from models.required_models import get_chat_model
from agents.state import AgentState, EvidenceGrade

logger = logging.getLogger(__name__)
model = get_chat_model()

async def evidence_grader_node(state: AgentState):
    docs = state.get("documents","")
    
    if not docs:
        return {"evidence_sufficient": False}

    query = state.get("search_query") or state["messages"][-1].content
    structured_llm = model.with_structured_output(EvidenceGrade)

    prompt = (
        f"User Query: {query}\n\nRetrieved Passages:\n{docs}\n\n"
        "Evaluate whether the retrieved passages contain relevant facts, background context, or direct information "
        "useful for answering the user's query.\n"
        "Be inclusive: if the passages contain partial facts, definitions, specifications, or related details "
        "that help address the prompt, mark is_sufficient as True."
    )
    try:
        res = await structured_llm.ainvoke([HumanMessage(content=prompt)])
        return {"evidence_sufficient": bool(res and res.is_sufficient)}
    except Exception:
        logger.exception("Evidence grading failed; treating retrieved evidence as insufficient.")
        return {"evidence_sufficient": True}

async def refine_node(state: AgentState):
    query = state.get("search_query") or state["messages"][-1].content
    retries = state.get("retry_count", 0) + 1

    original_question = state["messages"][-1].content
    prompt = (
        "Suggest one concise alternative search query for the user's indexed sources. "
        "Keep the original subject and requested facts; do not answer the question. "
        "Return only the query.\n"
        f"Original question: {original_question}\n"
        f"Previous search query: {query}"
    )
    res = await model.ainvoke([HumanMessage(content=prompt)])
    return {"search_query": res.content.strip(), "retry_count": retries}