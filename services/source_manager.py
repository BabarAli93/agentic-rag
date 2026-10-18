from uuid import uuid4
from urllib.parse import urlparse

from ingestion.loader import load_url, load_file
from ingestion.indexer import index_documents
from storage.qdrant import delete_source_vectors, get_chunks_by_source_id
from storage import source_registry as registry


def add_url_source(url: str, owner_id: str) -> dict:
    source_id = str(uuid4())
    parsed = urlparse(url)

    if (parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        ):
        raise ValueError("Please provide valid HTTP(s) URLs.")


    registry.add_source(
        source_id=source_id,
        owner_id=owner_id,
        source_type="url",
        source_name=url,
        source_uri=url
    )

    return _index_source(
        source_id=source_id,
        owner_id=owner_id,
        source_name=url,
        source_type="url",
        source_uri=url,
    )


def add_file_source(
        file_path: str,
        filename: str,
        owner_id: str
) -> dict:

    source_id = str(uuid4())
    suffix = filename.rsplit(".", 1)[-1].lower()

    if suffix not in {"pdf", "txt"}:
        raise ValueError("Only PDF and TXT files are supported.")

    registry.add_source(
        source_id=source_id,
        owner_id=owner_id,
        source_name=filename, 
        source_type=suffix
    )

    return _index_source(
        source_id=source_id,
        owner_id=owner_id,
        source_name=filename,
        source_type=suffix,
        file_path=file_path,
    )


def _index_source(
        *,
        source_id: str,
        owner_id: str,
        source_name: str,
        source_type: str,
        source_uri: str | None = None, 
        file_path: str | None = None
) -> dict:

    try: 
        registry.update_source(
            source_id, owner_id, status="indexing", error_message=None
        )

        if source_type == "url":
            documents = load_url(source_uri, source_id=source_id)
        else:
            documents = load_file(file_path, source_name, source_id)

        count = index_documents(documents=documents,
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
            error_message=None
        )

        return registry.get_source(source_id, owner_id)

    except Exception as e:
        registry.update_source(
            source_id=source_id,
            owner_id=owner_id,
            status="failed",
            error_message=str(e)[:1000],
        )
        raise


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
    source = registry.get_source(source_id, owner_id)

    if source is None:
        raise ValueError("Source not found.")

    # Do not erase the existing working vectors before a replacement
    # has been prepared successfully. See the re-indexing note below.
    return _reindex_existing_source(source, owner_id)

def _reindex_existing_source(source: dict, owner_id: str) -> dict:
    """
    Placeholder for safe, atomic re-indexing.

    File sources need their original file retained in durable storage.
    URL sources can be loaded again from source_uri.
    """
    raise NotImplementedError(
        "Implement staged replacement before enabling re-index."
    )