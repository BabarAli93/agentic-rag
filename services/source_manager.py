from uuid import uuid4
from urllib.parse import urlparse
import logging
import time
import hashlib
from pathlib import Path

from ingestion.loader import load_url, load_file, load_youtube
from ingestion.indexer import index_documents
from storage.qdrant import delete_source_vectors, get_chunks_by_source_id
from storage import source_registry as registry

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2

def add_url_source(url: str, owner_id: str) -> dict:
    source_id = str(uuid4())
    parsed = urlparse(url)

    if (parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password):
        raise ValueError("Please provide valid HTTP(s) URLs.")

    is_youtube = "youtube.com" in parsed.netloc or "youtu.be" in parsed.netloc
    source_type = "youtube" if is_youtube else "url"

    registry.add_source(
        source_id=source_id,
        owner_id=owner_id,
        source_type=source_type,
        source_name=url,
        source_uri=url
    )

    return _index_source_with_retries(
        source_id=source_id,
        owner_id=owner_id,
        source_name=url,
        source_type=source_type,
        source_uri=url,
    )


def add_file_source(
        file_path: str,
        filename: str,
        owner_id: str
) -> dict:

    source_id = str(uuid4())
    suffix = filename.rsplit(".", 1)[-1].lower()

    if suffix not in {"pdf", "txt", "md", "docx"}:
        raise ValueError("Only PDF, TXT, MD, and DOCX files are supported.")

    registry.add_source(
        source_id=source_id,
        owner_id=owner_id,
        source_name=filename, 
        source_type=suffix,
        source_uri=file_path
    )

    return _index_source_with_retries(
        source_id=source_id,
        owner_id=owner_id,
        source_name=filename,
        source_type=suffix,
        file_path=file_path,
    )


def _index_source_with_retries(
        *,
        source_id: str,
        owner_id: str,
        source_name: str,
        source_type: str,
        source_uri: str | None = None, 
        file_path: str | None = None
) -> dict:
    """Attempts ingestion and indexing with retries and backoff on failure."""
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try: 
            registry.update_source(
                source_id, owner_id, status="indexing", error_message=None
            )

            if source_type == "url":
                documents = load_url(source_uri, source_id=source_id)
            elif source_type == "youtube":
                documents = load_youtube(source_uri, source_id=source_id)
            else:
                documents = load_file(file_path, source_name, source_id)

            content_hash = _compute_content_hash(documents)

            count = index_documents(
                documents=documents,
                source_id=source_id, 
                source_name=source_name,
                source_type=source_type,
                owner_id=owner_id
            )
            if count == 0:
                raise ValueError("No indexable text was found.")

            registry.update_source(
                source_id=source_id,
                owner_id=owner_id,
                status="indexed",
                chunk_count=count,
                error_message=None,
                content_hash=content_hash
            )

            return registry.get_source(source_id, owner_id)

        except Exception as e:
            last_error = e
            logger.warning(f"Ingestion attempt {attempt}/{MAX_RETRIES} failed for source {source_id}: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS * attempt)

            
    # If all retries fail, mark source status as 'failed'
    error_msg = str(last_error)[:1000] if last_error else "Unknown ingestion error."
    registry.update_source(
        source_id=source_id,
        owner_id=owner_id,
        status="failed",
        error_message=error_msg,
    )
    raise last_error


def list_sources(owner_id: str) -> list[dict]:
    return registry.list_sources(owner_id)


def remove_source(source_id: str, owner_id: str) -> bool:
    source = registry.get_source(source_id, owner_id)

    if source is None:
        return False

    delete_source_vectors(source_id, owner_id)
    return registry.delete_source(source_id, owner_id)

def get_chunks_by_source(source_id: str, owner_id: str) -> list[dict]:
    return get_chunks_by_source_id(source_id=source_id, owner_id=owner_id)

        
def reindex_source(source_id: str, owner_id: str) -> dict:
    """Safe atomic re-indexing workflow for outdated sources."""
    source = registry.get_source(source_id, owner_id)

    if source is None:
        raise ValueError("Source not found.")

    return _reindex_existing_source(source, owner_id)

def _reindex_existing_source(source: dict, owner_id: str) -> dict:
    """
    - Performs safe atomic re-indexing: loads new content, indexes under the 
    ORIGINAL source_id only after successful parsing, and cleans up old vectors.
    
    - Incremental re-indexing: checks if content has changed via hash comparison.
    If unchanged, skips expensive re-embedding entirely.
    
    """
    source_id = source["source_id"]
    source_type = source["source_type"]
    source_uri = source["source_uri"]
    source_name = source["source_name"]
    existing_hash = source.get("content_hash")

    try:
        registry.update_source(source_id, owner_id, status="indexing", error_message=None)

        # 1. Load fresh content using the original source_id right away
        if source_type == "url":
            documents = load_url(source_uri, source_id=source_id)
        elif source_type == "youtube":
            documents = load_youtube(source_uri, source_id=source_id)
        else:
            # Load local files from permanent storage path
            if not source_uri or not Path(source_uri).exists():
                raise FileNotFoundError("Original file path no longer exists on disk.")
            documents = load_file(source_uri, source_name=source_name, source_id=source_id)

        new_hash = _compute_content_hash(documents)

        # 2. INCREMENTAL SHORT-CIRCUIT: If content hasn't changed, skip re-embedding!
        if existing_hash and new_hash == existing_hash:
            registry.update_source(source_id, owner_id, status="indexed", error_message=None)
            return registry.get_source(source_id, owner_id)

        # 3. Delete old vectors from Qdrant before writing new ones
        delete_source_vectors(source_id, owner_id)
        
        count = index_documents(
            documents=documents,
            source_id=source_id,
            source_name=source_name,
            source_type=source_type,
            owner_id=owner_id
        )

        if count == 0:
            raise ValueError("Re-indexing yielded no indexable text.")

        # 5. Update source registry metrics
        registry.update_source(
            source_id=source_id,
            owner_id=owner_id,
            status="indexed",
            chunk_count=count,
            error_message=None,
            content_hash=new_hash
        )

        return registry.get_source(source_id, owner_id)

    except Exception as e:
        registry.update_source(
            source_id=source_id,
            owner_id=owner_id,
            status="failed",
            error_message=str(e)[:1000]
        )
        raise

def _compute_content_hash(documents: list) -> str:
    """Computes a unified SHA-256 hash across all document pages/chunks."""
    hasher = hashlib.sha256()
    for doc in documents:
        hasher.update(doc.page_content.encode("utf-8"))
    return hasher.hexdigest()