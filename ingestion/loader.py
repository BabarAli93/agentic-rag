from pathlib import Path
from uuid import uuid4
from urllib.parse import urlparse

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    WebBaseLoader
)

SUPPORTED_EXTENSION = {
    ".pdf",
    ".txt"
}

def load_file(path: str, 
              source_name: str | None = None,
              source_id: str | None = None) -> list[Document]:

    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix == '.pdf':
        loader = PyPDFLoader(str(file_path))
    elif suffix == '.txt':
        loader = TextLoader(
            str(file_path),
            encoding="utf-8"
        )
    else:
        raise ValueError (
            f"Unsupported file type: {suffix}"
        )

    documents = loader.load()
    source_id = source_id or str(uuid4())
    original_name = source_name or file_path.name

    for document in documents:
        document.metadata.update(
            {
                "source_id": source_id,
                "source": original_name,
                "source_type": suffix.lstrip("."),
            }
        )

    return documents


def load_url(url: str, source_id: str | None = None) -> list[Document]:

    parsed = urlparse(url)

    if parsed.scheme not in {"http", "https"}:
         raise ValueError("Only HTTP(s) URLs are supported.")

    if not parsed.hostname:
         raise ValueError("Invalid website URL.")

    loader = WebBaseLoader(url)
    documents = loader.load()
    source_id = source_id or str(uuid4())

    for document in documents:
        document.metadata.update(
            {
                "source_id": source_id,
                "source": url,
                "source_type": "url"
                }
            )
    
    return documents