"""Read PDFs and split them into overlapping chunks that remember their page number."""

import re
from pathlib import Path

from pypdf import PdfReader


def load_pdf_pages(path: Path) -> list[dict]:
    """Return [{"page": 1, "text": "..."}, ...] for a PDF. Empty pages are skipped."""
    reader = PdfReader(str(path))
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append({"page": number, "text": text})
    return pages


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    """Split text into chunks of ~chunk_size characters with some overlap.

    Tries to cut at a sentence boundary so chunks read naturally.
    Chunking strategy is one of the first things to experiment with in Stage 2.
    """
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= chunk_size:
        return [text] if text else []

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = text.rfind(". ", start, end)
            if boundary > start + chunk_size // 2:
                end = boundary + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def pdf_to_chunks(path: Path, chunk_size: int, overlap: int) -> list[dict]:
    """Turn one PDF into chunk records ready to embed and index."""
    records = []
    for page in load_pdf_pages(path):
        for i, chunk in enumerate(chunk_text(page["text"], chunk_size, overlap)):
            records.append(
                {
                    "source": path.name,
                    "page": page["page"],
                    "chunk_index": i,
                    "content": chunk,
                }
            )
    return records
