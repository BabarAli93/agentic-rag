import os
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from flashrank import Ranker

# Define strict allowed modes
VALID_MODES = ("remote", "local", "openai")

def get_deployment_mode() -> str:
    """Reads and strictly validates DEPLOYMENT_MODE from .env"""
    mode = os.getenv("DEPLOYMENT_MODE", "remote").lower().strip()
    
    if mode not in VALID_MODES:
        raise ValueError(
            f"❌ Invalid DEPLOYMENT_MODE '{mode}' found in environment variables. "
            f"Allowed options are: {list(VALID_MODES)}"
        )
    return mode

def get_chat_model(temperature=0.0):
    mode = get_deployment_mode()
    
    if mode == "local":
        return ChatOllama(
            model=os.getenv("LOCAL_MODEL_NAME", "qwen2.5:0.5b"),
            temperature=temperature,
        )
    elif mode == "openai":
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY")
        )
    else: # remote
        print("Builing remote vLLM")
        return ChatOpenAI(
            model=os.getenv("VLLM_MODEL_NAME", "Qwen/Qwen3-32B"),
            temperature=temperature,
            base_url=os.getenv('VLLM_BASE_URL'),
            api_key=os.getenv("VLLM_API_KEY", "your-secure-token")
        )

def get_embedding_model():
    mode = get_deployment_mode()
    
    if mode == "local":
        return HuggingFaceEmbeddings(model_name=os.getenv("LOCAL_EMBED_MODEL", "BAAI/bge-large-en-v1.5"))
    elif mode == "openai":
        return OpenAIEmbeddings(model=os.getenv("OPENAI_EMBED_MODEL"), api_key=os.getenv("OPENAI_API_KEY"))
    else: # remote
        return OpenAIEmbeddings(
            model=os.getenv("VLLM_EMBED_MODEL", "BAAI/bge-large-en-v1.5"),
            base_url=os.getenv('VLLM_EMBEDDING_BASE_URL'),
            api_key=os.getenv("VLLM_API_KEY", "your-secure-token"),
            check_embedding_ctx_length=False
        )

def get_ranking_model():
    return Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/tmp/flashrank")