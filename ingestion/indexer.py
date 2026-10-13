from langchain_core.documents import Document

from storage.qdrant import get_vector_store
from ingestion.splitter import split_documents


def index_documents(documents: list[Document]) -> int:

    chunks = split_documents(documents)

    if not chunks:
        return 0

    vector_store = get_vector_store()

    vector_store.add_documents(chunks)

    return len(chunks)
