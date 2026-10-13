from pathlib import Path
from uuid import uuid4

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

def load_file(path: str) -> list[Document]:
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

    document_id = str(uuid4())

    for document in documents:
        document.metadata.update(
            {
                "document_id": document_id,
                "source": file_path.name,
                "source_type": suffix.lstrip("."),
            }
        )

    return documents


def load_url(url: str) -> list[Document]:

    loader = WebBaseLoader(url)

    documents = loader.load()

    document_id = str(uuid4())

    for document in documents:
            document.metadata.update(
                {
                    "document_id": document_id,
                    "source": url,
                    "source_type": "web"
                }
            )
    
    return documents