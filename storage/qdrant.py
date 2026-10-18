from qdrant_client import QdrantClient, models
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse

from models.embeddings import get_embeddings
from config import QDRANT_COLLECTION, QDRANT_URL


def get_qdrant_client() -> QdrantClient:
    """
    Returns an instance of QdrantClient for interacting with the Qdrant vector database.
    """
    return QdrantClient(url=QDRANT_URL)

def get_sparse_embeddings():
    """Returns local FastEmbed sparse model for keyword/BM25 representation."""
    return FastEmbedSparse(model_name="Qdrant/bm25")

def get_vector_store() -> QdrantVectorStore:
    """
    Returns a QdrantVectorStore configured for hybrid (dense + sparse) retrieval,
    automatically creating the collection if it doesn't exist.
    """
    qdrant_client = get_qdrant_client()
    embeddings = get_embeddings()
    sparse_embeddings = get_sparse_embeddings()

    # Automatically create the collection with named dense and sparse configs if missing
    if not qdrant_client.collection_exists(QDRANT_COLLECTION):
        qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config={
                "dense": models.VectorParams(
                    size=len(embeddings.embed_query("test")),
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False)
                )
            },
        )

    return QdrantVectorStore(
        client=qdrant_client,
        collection_name=QDRANT_COLLECTION,
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name="dense",
        sparse_vector_name="sparse",
    )
def delete_source_vectors(source_id: str, 
                          owner_id: str | None = None) -> None:
    """Delete all indexed chunks belonging to one source."""
    
    conditions=[
        models.FieldCondition(
            key="metadata.source_id",
            match=models.MatchValue(value=source_id),
        )
    ]

    if owner_id is not None:
        conditions.append(
            models.FieldCondition(
                key="metadata.owner_id",
                match=models.MatchValue(value=owner_id),
            )
        )

    client = get_qdrant_client()

    client.delete(
        collection_name=QDRANT_COLLECTION,
        points_selector=models.FilterSelector(
            filter=models.Filter(must=conditions)
        ),
        wait=True,
    )


def get_chunks_by_source_id(source_id: str, owner_id: str) -> list[dict]:
    """
    Retrieves all text chunks belonging to a specific source directly from Qdrant.
    """

    client= get_qdrant_client()

    q_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="metadata.source_id",
                match=models.MatchValue(value=source_id)
            ),
            models.FieldCondition(
                key="metadata.owner_id",
                match=models.MatchValue(value=owner_id)
            )
        ]
    )

    points, _ = client.scroll(
        collection_name=QDRANT_COLLECTION,
        scroll_filter=q_filter,
        with_payload=True, 
        with_vectors=False,
        limit=500,
    )

    chunks = []

    for point in points:
        payload = point.payload or {}
        text = payload.get("page_content") or payload.get("text") or payload.get("metadata", {}).get("text", str(payload))

        chunks.append({
            "chunk_id": point.id,
            "text": text,
            "metadata": payload
        })

    return chunks