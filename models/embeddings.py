from langchain_openai import OpenAIEmbeddings

def get_embeddings():
    """
    Returns an instance of OpenAIEmbeddings for generating embeddings.
    """
    return OpenAIEmbeddings(
        model="text-embedding-3-small"
    )