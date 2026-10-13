from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore

from models.embeddings import get_embeddings
from config import QDRANT_COLLECTION, QDRANT_URL


def get_qdrant_client() -> QdrantClient:
    """
    Returns an instance of QdrantClient for interacting with the Qdrant vector database.
    """
    return QdrantClient(url=QDRANT_URL)

def get_vector_store() -> QdrantVectorStore:
    """
    Returns an instance of QdrantVectorStore for storing and retrieving vector embeddings.
    """
    qdrant_client = get_qdrant_client()
    embeddings = get_embeddings()

    if not qdrant_client.collection_exists(QDRANT_COLLECTION):
        qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config={
                "size": len(embeddings.embed_query("test")),
                "distance": "Cosine",
            },
        )

    return QdrantVectorStore(
        client=qdrant_client,
        collection_name=QDRANT_COLLECTION,
        embedding=embeddings
    )