from storage.qdrant import get_vector_store

def get_retriever():
    
    vector_store = get_vector_store()

    return vector_store.as_retriever(
        search_kwargs= {
            "k": 5,
        }
    )