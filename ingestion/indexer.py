from uuid import uuid4

from langchain_core.documents import Document

from storage.qdrant import get_vector_store
from ingestion.splitter import split_documents


def index_documents(
        documents: list[Document],
        source_id: str | None = None,
        source_name: str | None = None,
        source_type: str | None = None,
        owner_id: str | None = None,) -> int:

    """
    Split documents, attach source metadata, and index chunks.

    Optional arguments preserve compatibility with existing callers.
    Source management should always supply a stable source_id.
    """

    if not documents:
        return 0

    source_id = source_id or str(uuid4())

    chunks = split_documents(documents)

    if not chunks:
        return 0

    for chunk in chunks:
        chunk.metadata = {
            **chunk.metadata,
            "source_id": source_id,
        }

        if source_name is not None:
            chunk.metadata["source_name"] = source_name

        if source_type is not None:
            chunk.metadata["source_type"] = source_type

        if owner_id is not None:
            chunk.metadata["owner_id"] = owner_id


    vector_store = get_vector_store()
    vector_store.add_documents(chunks)

    return len(chunks)
