

import sys
from pathlib import Path

from src.rag.config import get_settings
from src.rag.embeddings import embed_texts, get_openai_client
from src.rag.loader import pdf_to_chunks
from src.rag.vector_store import get_store

DEFAULT_FOLDER = Path("data/sample_docs")


def upload_to_blob(settings, pdf_paths: list[Path]) -> None:
    """Keep a copy of the raw documents in Azure Blob Storage (skipped if not configured)."""
    if not settings.storage_connection_string:
        return

    from azure.storage.blob import BlobServiceClient

    service = BlobServiceClient.from_connection_string(settings.storage_connection_string)
    container = service.get_container_client(settings.storage_container)
    if not container.exists():
        container.create_container()
    for path in pdf_paths:
        with open(path, "rb") as f:
            container.upload_blob(name=path.name, data=f, overwrite=True)
        print(f"  uploaded to blob: {path.name}")


def main() -> None:
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_FOLDER
    pdf_paths = sorted(folder.glob("*.pdf"))
    if not pdf_paths:
        print(f"No PDF files found in {folder}. Add some and run again.")
        return

    settings = get_settings()
    client = get_openai_client(settings)
    store = get_store(settings)

    print(f"Provider: {settings.provider} | Vector store: {settings.vector_store}")
    print(f"Found {len(pdf_paths)} PDF(s).")
    upload_to_blob(settings, pdf_paths)

    for path in pdf_paths:
        chunks = pdf_to_chunks(path, settings.chunk_size, settings.chunk_overlap)
        if not chunks:
            print(f"  {path.name}: no extractable text (scanned PDF? see Stage 2 notes)")
            continue
        vectors = embed_texts(client, settings, [c["content"] for c in chunks])
        count = store.upload_chunks(chunks, vectors)
        print(f"  {path.name}: indexed {count} chunks")

    print("Done.")


if __name__ == "__main__":
    main()
