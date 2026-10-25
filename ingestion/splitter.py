from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

def split_documents(documents: list[Document]) -> list[Document]:
    """This function will receive a list of documents and chunk them """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 2000,
        chunk_overlap = 400
    )

    chunks = splitter.split_documents(documents)

    for index, chunk in enumerate(chunks):
        chunk.metadata['chunk_index'] = index

    return chunks