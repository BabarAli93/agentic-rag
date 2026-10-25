import logging

from langchain_core.messages import HumanMessage
from agents.state import AgentState, RouterDecision
from models.required_models import get_chat_model

logger = logging.getLogger(__name__)
model = get_chat_model()

async def router_node(state: AgentState):
    last_message = state["messages"][-1].content
    structured_llm = model.with_structured_output(RouterDecision)

    prompt = (
        "You are routing a source-grounded document question-answering assistant. "
        "Every factual question must use the user's indexed documents, even when it "
        "asks for a common definition or sounds answerable from general knowledge. "
        "Choose false only for greetings, thanks, or non-factual conversation.\n\n"
        f"User message:\n{last_message}"
    )
    try:
        res = await structured_llm.ainvoke([HumanMessage(content=prompt)])
        needs_lookup = res.needs_lookup if res else True
    except Exception:
        logger.exception("Source-routing decision failed; defaulting to retrieval.")
        needs_lookup = True

    return {"needs_lookup": needs_lookup, "retry_count": 0}


async def general_chat_node(state: AgentState):
    res = await model.ainvoke(state["messages"])
    return {"messages": [res]}
