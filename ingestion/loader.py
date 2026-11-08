from pathlib import Path
from uuid import uuid4
from urllib.parse import urlparse, parse_qs

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
    Docx2txtLoader,
)

from youtube_transcript_api import YouTubeTranscriptApi

SUPPORTED_EXTENSION = {
    ".pdf",
    ".txt",
    ".md",
    ".docx"
}

def load_file(path: str, 
              source_name: str | None = None,
              source_id: str | None = None) -> list[Document]:

    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix == '.pdf':
        loader = PyPDFLoader(str(file_path))
    elif suffix in {'.txt', '.md'}:
        loader = TextLoader(str(file_path), encoding="utf-8")
    elif suffix == '.docx':
        loader = Docx2txtLoader(str(file_path))
    else:
        raise ValueError (f"Unsupported file type: {suffix}")

    documents = loader.load()
    source_id = source_id or str(uuid4())
    original_name = source_name or file_path.name

    for document in documents:
        document.metadata.update(
            {
                "source_id": source_id,
                "source": original_name,
                "source_type": suffix.lstrip("."),
            }
        )

    return documents


def load_url(url: str, source_id: str | None = None) -> list[Document]:

    parsed = urlparse(url)

    if parsed.scheme not in {"http", "https"}:
         raise ValueError("Only HTTP(s) URLs are supported.")

    if not parsed.hostname:
         raise ValueError("Invalid website URL.")

    loader = WebBaseLoader(url)
    documents = loader.load()
    source_id = source_id or str(uuid4())

    for document in documents:
        document.metadata.update(
            {
                "source_id": source_id,
                "source": url,
                "source_type": "url",
                }
            )
    
    return documents


def load_youtube(url: str, source_id: str | None = None) -> list[Document]:
    """Loads transcripts directly using the updated youtube-transcript-api syntax."""
    try:
        parsed_url = urlparse(url)
        if parsed_url.hostname == 'youtu.be':
            video_id = parsed_url.path[1:]
        elif parsed_url.hostname in {'www.youtube.com', 'youtube.com'}:
            if parsed_url.path == '/watch':
                video_id = parse_qs(parsed_url.query).get('v', [None])[0]
            elif parsed_url.path.startswith('/embed/') or parsed_url.path.startswith('/v/'):
                video_id = parsed_url.path.split('/')[2]
            else:
                raise ValueError("Could not parse video ID from YouTube URL path.")
        else:
            raise ValueError("Invalid YouTube URL domain.")

        if not video_id:
            raise ValueError("Missing YouTube video ID parameter.")

        # Instantiate the client and use .fetch() instead of the old get_transcript() method
        ytt_api = YouTubeTranscriptApi()
        fetched_transcript = ytt_api.fetch(video_id)
        
        # Join text snippets from the fetched transcript object
        full_text = " ".join([snippet.text for snippet in fetched_transcript])

    except Exception as e:
        raise ValueError(f"Could not load YouTube transcript. Ensure captions/subtitles are enabled on the video. Error: {e}")

    source_id = source_id or str(uuid4())

    document = Document(
        page_content=full_text,
        metadata={
            "source_id": source_id,
            "source": url,
            "source_type": "youtube",
        }
    )

    return [document]