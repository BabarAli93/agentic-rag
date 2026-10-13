import os
import tempfile

import chainlit as cl

from ingestion.loader import (
    load_file,
    load_url
)
from ingestion.indexer import index_documents
from agents.rag_agent import agent

@cl.on_chat_start
async def start():
    await cl.Message(
        content=(
            "## Agentic RAG\n\n"
            "Upload PDF/TXT files or provide website URLs, "
            "then ask questions about them."
        )
    ).send()


@cl.on_message
async def main(message: cl.Message):

    if message.elements:
        await process_uploaded_files(message.elements)

    if not message.content.strip():
        return

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": message.content,
                }
            ]
        }
    )

    final_message =  result["messages"][-1]

    await cl.Message(
        content=final_message.content
    ).send()


async def process_uploaded_files(elements):

    for element in elements:

        suffix = os.path.splitext(element.name)[1].lower()

        if suffix not in [".pdf", ".txt"]:
            await cl.Message(
                content=(
                    f"Unsupported file type: {element.name}"
                )
            ).send()

            continue

        with tempfile.NamedTemporaryFile(
            suffix=suffix, delete=False
        ) as temp_file:

            with open(element.path, "rb") as source:

                temp_file.write(source.read())

            temp_path = temp_file.name

        documents = load_file(temp_path)

        count = index_documents(documents=documents)

        await cl.Message(
            content=(
                f"Indexed **{element.name}** ({count} chunks)."
            )
        ).send()
