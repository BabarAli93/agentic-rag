from ingestion.loader import load_file, load_url
from ingestion.indexer import index_documents


pdf_docs = load_file(
    "./documents/3700410.3702126.pdf"
)

print(f"Loaded {len(pdf_docs)} pdf documents.")

counts = index_documents(pdf_docs)

print(
    f"Indexed {counts} chunks"
)