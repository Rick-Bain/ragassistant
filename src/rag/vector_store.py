"""Vector stores. Two interchangeable backends with the same three methods:

    store = get_store(settings)
    store.create_index()                      # one-time setup
    store.upload_chunks(chunks, vectors)      # index chunk records + their embeddings
    store.search(query, query_vector)         # hybrid (keyword + vector) -> top_k chunks

  - LocalStore:       JSON file on disk, runs anywhere, no accounts needed.
  - AzureSearchStore: Azure AI Search (HNSW vector index + BM25 text, fused by the service).
"""

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Protocol

import numpy as np

from .config import Settings


class VectorStore(Protocol):
    def create_index(self) -> str: ...
    def upload_chunks(self, chunks: list[dict], vectors: list[list[float]]) -> int: ...
    def search(self, query: str, query_vector: list[float]) -> list[dict]: ...


def _doc_id(source: str, page: int, chunk_index: int) -> str:
    safe = re.sub(r"[^A-Za-z0-9_=-]", "_", source)
    return f"{safe}-p{page}-c{chunk_index}"


# --------------------------------------------------------------------------------------
# Local store
# --------------------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _bm25_scores(query_tokens: list[str], doc_tokens: list[list[str]], k1=1.5, b=0.75) -> list[float]:
    """Plain BM25 keyword scoring (the same idea Azure AI Search uses for the text side)."""
    n_docs = len(doc_tokens)
    avg_len = (sum(len(d) for d in doc_tokens) / n_docs) or 1.0
    doc_freq: Counter = Counter()
    for tokens in doc_tokens:
        doc_freq.update(set(tokens))

    scores = []
    for tokens in doc_tokens:
        tf = Counter(tokens)
        score = 0.0
        for term in set(query_tokens):
            if term not in tf:
                continue
            idf = math.log(1 + (n_docs - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
            freq = tf[term]
            score += idf * freq * (k1 + 1) / (freq + k1 * (1 - b + b * len(tokens) / avg_len))
        scores.append(score)
    return scores


class LocalStore:
    """Chunks + embeddings in one JSON file. Fine for up to a few thousand chunks."""

    RRF_K = 60  # reciprocal rank fusion constant (same default Azure AI Search uses)

    def __init__(self, settings: Settings):
        self.settings = settings
        self.path = Path(settings.local_index_path)

    def _meta(self) -> dict:
        return {
            "embedding_model": self.settings.embedding_model,
            "dimensions": self.settings.embedding_dimensions,
        }

    def _load(self) -> dict:
        if not self.path.exists():
            return {"meta": self._meta(), "docs": {}}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _save(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        tmp.replace(self.path)

    def create_index(self) -> str:
        data = self._load() if self.path.exists() else None
        if data and data["meta"] == self._meta():
            return f"Local index at {self.path} is ready ({len(data['docs'])} chunks already stored)."
        reset = " (embedding model changed, old data discarded; re-run ingest)" if data else ""
        self._save({"meta": self._meta(), "docs": {}})
        return f"Created local index at {self.path}{reset}."

    def upload_chunks(self, chunks: list[dict], vectors: list[list[float]]) -> int:
        data = self._load()
        if data["meta"] != self._meta():
            raise RuntimeError(
                f"The local index was built with {data['meta']} but you are now using {self._meta()}. "
                "Vectors from different embedding models can't be mixed. "
                "Run `python -m scripts.create_index` to reset it, then ingest again."
            )
        for chunk, vector in zip(chunks, vectors):
            doc_id = _doc_id(chunk["source"], chunk["page"], chunk["chunk_index"])
            data["docs"][doc_id] = {**chunk, "embedding": vector}
        self._save(data)
        return len(chunks)

    def search(self, query: str, query_vector: list[float]) -> list[dict]:
        data = self._load()
        docs = list(data["docs"].values())
        if not docs:
            return []
        if data["meta"] != self._meta():
            raise RuntimeError(
                "The local index was built with a different embedding model. "
                "Reset it with `python -m scripts.create_index` and ingest again."
            )

        # Vector side: cosine similarity
        matrix = np.array([d["embedding"] for d in docs], dtype=np.float32)
        q = np.array(query_vector, dtype=np.float32)
        sims = matrix @ q / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(q) + 1e-9)
        vector_order = np.argsort(-sims)

        # Keyword side: BM25
        kw_scores = _bm25_scores(_tokenize(query), [_tokenize(d["content"]) for d in docs])
        keyword_order = [i for i in np.argsort(-np.array(kw_scores)) if kw_scores[i] > 0]

        # Fuse the two rankings with reciprocal rank fusion
        fused: dict[int, float] = {}
        for ranking in (vector_order, keyword_order):
            for rank, idx in enumerate(ranking):
                fused[int(idx)] = fused.get(int(idx), 0.0) + 1.0 / (self.RRF_K + rank + 1)

        top = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[: self.settings.top_k]
        return [
            {
                "content": docs[i]["content"],
                "source": docs[i]["source"],
                "page": docs[i]["page"],
                "score": score,
            }
            for i, score in top
        ]


# --------------------------------------------------------------------------------------
# Azure AI Search store
# --------------------------------------------------------------------------------------

class AzureSearchStore:
    VECTOR_PROFILE = "default-vector-profile"
    VECTOR_ALGORITHM = "default-hnsw"

    def __init__(self, settings: Settings):
        self.settings = settings

    def _credential(self):
        from azure.core.credentials import AzureKeyCredential

        return AzureKeyCredential(self.settings.search_api_key)

    def _search_client(self):
        from azure.search.documents import SearchClient

        return SearchClient(self.settings.search_endpoint, self.settings.search_index, self._credential())

    def create_index(self) -> str:
        """Create (or update) the index with a text field and a vector field."""
        from azure.search.documents.indexes import SearchIndexClient
        from azure.search.documents.indexes.models import (
            HnswAlgorithmConfiguration,
            SearchableField,
            SearchField,
            SearchFieldDataType,
            SearchIndex,
            SimpleField,
            VectorSearch,
            VectorSearchProfile,
        )

        s = self.settings
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(name="content", type=SearchFieldDataType.String),
            SimpleField(name="source", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="page", type=SearchFieldDataType.Int32, filterable=True),
            SimpleField(name="chunk_index", type=SearchFieldDataType.Int32),
            SearchField(
                name="embedding",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=s.embedding_dimensions,
                vector_search_profile_name=self.VECTOR_PROFILE,
            ),
        ]
        vector_search = VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name=self.VECTOR_ALGORITHM)],
            profiles=[
                VectorSearchProfile(
                    name=self.VECTOR_PROFILE, algorithm_configuration_name=self.VECTOR_ALGORITHM
                )
            ],
        )
        index = SearchIndex(name=s.search_index, fields=fields, vector_search=vector_search)
        SearchIndexClient(s.search_endpoint, self._credential()).create_or_update_index(index)
        return f"Azure AI Search index '{s.search_index}' is ready ({s.embedding_dimensions} dimensions)."

    def upload_chunks(self, chunks: list[dict], vectors: list[list[float]]) -> int:
        client = self._search_client()
        documents = [
            {
                "id": _doc_id(c["source"], c["page"], c["chunk_index"]),
                "content": c["content"],
                "source": c["source"],
                "page": c["page"],
                "chunk_index": c["chunk_index"],
                "embedding": vector,
            }
            for c, vector in zip(chunks, vectors)
        ]
        uploaded = 0
        for i in range(0, len(documents), 100):
            result = client.upload_documents(documents=documents[i : i + 100])
            uploaded += sum(1 for r in result if r.succeeded)
        return uploaded

    def search(self, query: str, query_vector: list[float]) -> list[dict]:
        """Hybrid retrieval: BM25 keyword + vector similarity, fused by Azure AI Search."""
        from azure.search.documents.models import VectorizedQuery

        s = self.settings
        vector_query = VectorizedQuery(
            vector=query_vector, k_nearest_neighbors=s.top_k, fields="embedding"
        )
        results = self._search_client().search(
            search_text=query,
            vector_queries=[vector_query],
            select=["content", "source", "page"],
            top=s.top_k,
        )
        return [
            {
                "content": r["content"],
                "source": r["source"],
                "page": r["page"],
                "score": r["@search.score"],
            }
            for r in results
        ]


def get_store(settings: Settings) -> VectorStore:
    if settings.vector_store == "azure_search":
        return AzureSearchStore(settings)
    return LocalStore(settings)
