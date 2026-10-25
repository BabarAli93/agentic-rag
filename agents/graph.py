import os
from contextlib import asynccontextmanager
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agents.state import AgentState
from agents.nodes.router import router_node, general_chat_node
from agents.nodes.retrieval import rewriter_node, retrieval_and_rerank_node
from agents.nodes.grader import refine_node, evidence_grader_node
from agents.nodes.generator import generator_node, grounding_checker_node, abstain_node

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "storage", "checkpoints.sqlite")

retries = 2

@asynccontextmanager
async def get_checkpointer_cm():
    """Async context manager for SQLite checkpointer to ensure thread safety per request."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(DB_PATH) as checkpointer:
        yield checkpointer


# --- ROUTING CONDITIONS ---

def route_from_router(state: AgentState):
    return "rewriter" if state.get("needs_lookup", True) else "general_chat"

def route_from_grader(state: AgentState):
    if state.get("evidence_sufficient"):
        return "generator"
    return "refine" if state.get("retry_count", 0) < retries else "abstain"

def route_from_grounding(state: AgentState):
    if state.get("grounded"):
        return END
    return "refine" if state.get("retry_count", 0) < retries else "abstain"


# --- AGENT BUILDER ---
async def build_agent(owner_id: str, checkpointer=None):
    builder = StateGraph(AgentState)

    # Add Notes
    builder.add_node("router", router_node)
    builder.add_node("general_chat", general_chat_node)
    builder.add_node("rewriter", rewriter_node)
    builder.add_node("retrieval", retrieval_and_rerank_node)
    builder.add_node("grader", evidence_grader_node)
    builder.add_node("refine", refine_node)
    builder.add_node("generator", generator_node)
    builder.add_node("grounding", grounding_checker_node)
    builder.add_node("abstain", abstain_node)


    # Set the Edges and Entry
    builder.set_entry_point("router")
    builder.add_conditional_edges("router", route_from_router, {"rewriter": "rewriter", "general_chat": "general_chat"})
    builder.add_edge("general_chat", END)
    builder.add_edge("rewriter", "retrieval")
    builder.add_edge("retrieval", "grader")
    builder.add_conditional_edges("grader", route_from_grader, {"generator": "generator", "refine": "refine", "abstain": "abstain"})
    builder.add_edge("refine", "retrieval")
    builder.add_edge("generator", "grounding")
    builder.add_conditional_edges(
        "grounding",
        route_from_grounding,
        {END: END, "refine": "refine", "abstain": "abstain"},
    )
    builder.add_edge("abstain", END)

    return builder.compile(checkpointer=checkpointer)