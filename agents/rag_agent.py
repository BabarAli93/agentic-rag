from langchain_core.tools import tool
from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model

from retrieval.retriever import get_retriever

retriever = get_retriever()

#@tool(parse_docstring=True)
@tool
def search_knowledge_base(query:str) -> str:
    """Search the user's indexed documents for relevant information.

    Args:
        query: The question or information to search for.

    Returns:
        Relevant document passages with their sources.
    """

    documents = retriever.invoke(query)

    if not documents:
        return "No relevant documents were found."

    results = []

    for index, document in enumerate(documents):

        source = document.metadata.get(
            "source", "Unknown"
        )    

        source_type = document.metadata.get(
            "source_type", "Unknown"
        )

        results.append(
            f"""
--- Result {index + 1} ---
Source: {source}
Type: {source_type}

{document.page_content}
"""
        )

        return "\n".join(results)


model = init_chat_model(
    model="gpt-4o-mini"
)

SYSTEM_PROMPT = """
You are a helpful knowledge-base assistant.

Answer questions using the user's indexed documents.

When the answer requires information from the knowledge base:

1. Use search_knowledge_base.
2. Examine the retrieved sources.
3. If the information is insufficient, perform another search using a more specific query.
4. Base factual claims about the user's documents on retrieved information.
5. Mention the relevant source when answering.

Do not invent information that is not supported by the retrieved
documents.
"""

agent = create_deep_agent(
    model=model,
    tools=[search_knowledge_base],
    system_prompt=SYSTEM_PROMPT,
)