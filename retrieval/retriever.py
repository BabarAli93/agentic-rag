from storage.qdrant import get_vector_store
from qdrant_client import models

def get_retriever(owner_id: str | None = None):

    """
    Returns a retriever configured with tenant filtering and higher chunk recall
    to prepare for reranking.
    """
    
    vector_store = get_vector_store()

    # Fetch more initial candidates (e.g., 15) for the reranker to prune down
    search_kwargs= {"k": 15}

    if owner_id is not None:
        search_kwargs["filter"] = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.owner_id",
                    match=models.MatchValue(value=owner_id)
                )
            ]
        )

    return vector_store.as_retriever(search_kwargs=search_kwargs)