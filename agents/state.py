from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


# --- STATE DEFINITION ---
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    owner_id: str
    needs_lookup: bool
    search_query: str
    documents: str
    evidence_sufficient: bool
    grounded: bool
    generation: str
    retry_count: int


class RouterDecision(BaseModel):
    needs_lookup: bool = Field(description="True for factual questions that should be answered from the user's sources. False only for greetings or non-factual conversational messages.")

class EvidenceGrade(BaseModel):
    is_sufficient: bool = Field(description="True if retrieved passages contain facts necessary to answer the prompt accurately.")
    reason: str = Field(description="Brief explanation of why evidence is or is not sufficient.")

class GroundingGrade(BaseModel):
    is_grounded: bool = Field(description="True if all claims in answer are directly supported by context and citations are present.")