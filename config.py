import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "agentic_rag")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables.")
