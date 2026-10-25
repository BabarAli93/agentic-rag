import logging

from langchain_core.messages import HumanMessage, AIMessage
from models.required_models import get_chat_model
from agents.state import AgentState, GroundingGrade

logger = logging.getLogger(__name__)
model = get_chat_model()

from langchain_core.messages import AIMessage, HumanMessage
from models.required_models import get_chat_model
from agents.state import AgentState

model = get_chat_model()

async def generator_node(state: AgentState):
    docs = state.get("documents", "")
    user_msg = state["messages"][-1].content

    prompt = (
        "You are an expert source-grounded knowledge assistant. "
        "Provide a clear, detailed, and well-structured answer to the user's question "
        "using ONLY facts explicitly supported by the context excerpts below.\n\n"
        "CITATION RULES:\n"
        "- Every factual statement or claim MUST include an inline citation badge using the exact source name from the excerpt, formatted strictly as:.\n"
        "- Do not invent citations or omit source markers.\n\n"
        f"Context Excerpts:\n{docs}\n\nUser Question: {user_msg}"
    )

    res = await model.ainvoke([HumanMessage(content=prompt)])
    
    # Return as messages list so LangGraph standard stream catches it instantly
    return {"messages": [AIMessage(content=res.content)]}


async def grounding_checker_node(state: AgentState):
    gen = state.get("generation", "")
    docs = state.get("documents", "")

    structured_llm = model.with_structured_output(GroundingGrade)

    prompt = (
        "Check the answer against the provided context. Return is_grounded=true "
        "if the core claims are supported by the excerpts and include source citations. "
        "Return false only if there are major unsupported claims or clear hallucinations.\n\n"
        f"Context:\n{docs}\n\nAnswer:\n{gen}"
    )
    try:
        res = await structured_llm.ainvoke([HumanMessage(content=prompt)])
        is_grounded = bool(res and res.is_grounded)
    except Exception:
        logger.exception("Grounding check failed; rejecting the generated answer.")
        is_grounded = False

    if not is_grounded:
        return {"grounded": False}
    return {"grounded": True, "messages": [AIMessage(content=gen)]}


async def abstain_node(state: AgentState):
    msg = "I checked your indexed sources, but I couldn't find sufficient information to answer your question accurately."
    return {"messages": [AIMessage(content=msg)]}
