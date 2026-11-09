import dataclasses

import pytest

from src.rag.config import Settings
from src.rag.vector_store import LocalStore, get_store


def make_settings(tmp_path, **overrides) -> Settings:
    base = dict(
        provider="ollama",
        vector_store="local",
        chat_model="chat",
        embedding_model="embed-a",
        embedding_dimensions=3,
        openai_api_key="ollama",
        openai_base_url=None,
        azure_openai_endpoint=None,
        azure_openai_api_key=None,
        azure_openai_api_version="2024-10-21",
        search_endpoint=None,
        search_api_key=None,
        search_index="docs-index",
        local_index_path=str(tmp_path / "index.json"),
        chunk_size=1000,
        chunk_overlap=150,
        top_k=2,
        storage_connection_string=None,
        storage_container="documents",
    )
    base.update(overrides)
    return Settings(**base)


CHUNKS = [
    {"source": "policy.pdf", "page": 1, "chunk_index": 0, "content": "Employees get 20 days of annual leave."},
    {"source": "policy.pdf", "page": 2, "chunk_index": 0, "content": "Expense reports are due within 30 days."},
    {"source": "manual.pdf", "page": 5, "chunk_index": 0, "content": "Reset the router by holding the button."},
]
VECTORS = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def test_get_store_picks_local(tmp_path):
    assert isinstance(get_store(make_settings(tmp_path)), LocalStore)


def test_upload_then_vector_search(tmp_path):
    store = LocalStore(make_settings(tmp_path))
    store.create_index()
    assert store.upload_chunks(CHUNKS, VECTORS) == 3
    # query text matches nothing by keyword, so only the vector side decides
    results = store.search("zzz", [0.0, 0.95, 0.05])
    assert results[0]["source"] == "policy.pdf" and results[0]["page"] == 2


def test_keyword_side_contributes(tmp_path):
    store = LocalStore(make_settings(tmp_path))
    store.create_index()
    store.upload_chunks(CHUNKS, VECTORS)
    # vector points at the router chunk, but the keyword "annual leave" is only in the policy chunk
    results = store.search("annual leave", [0.0, 0.0, 1.0])
    sources_pages = {(r["source"], r["page"]) for r in results}
    assert ("policy.pdf", 1) in sources_pages and ("manual.pdf", 5) in sources_pages


def test_top_k_respected_and_empty_index(tmp_path):
    store = LocalStore(make_settings(tmp_path))
    assert store.search("anything", [1.0, 0.0, 0.0]) == []
    store.create_index()
    store.upload_chunks(CHUNKS, VECTORS)
    assert len(store.search("leave", [1.0, 0.0, 0.0])) == 2


def test_reupload_is_idempotent(tmp_path):
    store = LocalStore(make_settings(tmp_path))
    store.create_index()
    store.upload_chunks(CHUNKS, VECTORS)
    store.upload_chunks(CHUNKS, VECTORS)
    assert len(store._load()["docs"]) == 3


def test_changing_embedding_model_is_blocked_then_reset(tmp_path):
    store = LocalStore(make_settings(tmp_path))
    store.create_index()
    store.upload_chunks(CHUNKS, VECTORS)

    other = LocalStore(make_settings(tmp_path, embedding_model="embed-b"))
    with pytest.raises(RuntimeError, match="different embedding model|can't be mixed"):
        other.upload_chunks(CHUNKS, VECTORS)
    with pytest.raises(RuntimeError):
        other.search("leave", [1.0, 0.0, 0.0])

    message = other.create_index()  # reset
    assert "discarded" in message
    assert other.upload_chunks(CHUNKS, VECTORS) == 3
