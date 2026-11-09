"""Central configuration. All values come from environment variables (.env locally).

Two independent switches:

PROVIDER (where chat + embedding models come from)
  - "ollama": local models served by Ollama (free, offline)        <- default
  - "openai": the OpenAI API (or any OpenAI-compatible endpoint via OPENAI_BASE_URL)
  - "azure":  Azure OpenAI

VECTOR_STORE (where chunks are stored and searched)
  - "local":        a JSON file on disk with hybrid (keyword + vector) search   <- default
  - "azure_search": Azure AI Search
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

PROVIDERS = ("ollama", "openai", "azure")
VECTOR_STORES = ("local", "azure_search")


def _required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Missing environment variable {name}. Copy .env.example to .env and fill it in."
        )
    return value


@dataclass(frozen=True)
class Settings:
    provider: str
    vector_store: str

    # Models. For Azure these are *deployment names*; otherwise model names.
    chat_model: str
    embedding_model: str
    embedding_dimensions: int

    # OpenAI-style client (used for "openai" and "ollama")
    openai_api_key: str | None
    openai_base_url: str | None

    # Azure OpenAI (provider == "azure")
    azure_openai_endpoint: str | None
    azure_openai_api_key: str | None
    azure_openai_api_version: str

    # Azure AI Search (vector_store == "azure_search")
    search_endpoint: str | None
    search_api_key: str | None
    search_index: str

    # Local vector store (vector_store == "local")
    local_index_path: str

    # Chunking / retrieval
    chunk_size: int
    chunk_overlap: int
    top_k: int

    # Blob storage (optional)
    storage_connection_string: str | None
    storage_container: str


def get_settings() -> Settings:
    provider = os.getenv("PROVIDER", "ollama").strip().lower()
    vector_store = os.getenv("VECTOR_STORE", "local").strip().lower()

    if provider not in PROVIDERS:
        raise RuntimeError(f"PROVIDER must be one of {PROVIDERS}, got '{provider}'.")
    if vector_store not in VECTOR_STORES:
        raise RuntimeError(f"VECTOR_STORE must be one of {VECTOR_STORES}, got '{vector_store}'.")

    openai_api_key = openai_base_url = None
    azure_endpoint = azure_key = None

    if provider == "ollama":
        # Ollama exposes an OpenAI-compatible API, so we reuse the OpenAI client.
        openai_api_key = "ollama"  # required by the client, ignored by Ollama
        openai_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        chat_model = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:3b")
        embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
        default_dims = "768"  # nomic-embed-text
    elif provider == "openai":
        openai_api_key = _required("OPENAI_API_KEY")
        openai_base_url = os.getenv("OPENAI_BASE_URL") or None
        chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        default_dims = "1536"
    else:  # azure
        azure_endpoint = _required("AZURE_OPENAI_ENDPOINT")
        azure_key = _required("AZURE_OPENAI_API_KEY")
        chat_model = _required("AZURE_OPENAI_CHAT_DEPLOYMENT")
        embedding_model = _required("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
        default_dims = "1536"

    search_endpoint = search_key = None
    if vector_store == "azure_search":
        search_endpoint = _required("AZURE_SEARCH_ENDPOINT")
        search_key = _required("AZURE_SEARCH_API_KEY")

    return Settings(
        provider=provider,
        vector_store=vector_store,
        chat_model=chat_model,
        embedding_model=embedding_model,
        embedding_dimensions=int(os.getenv("EMBEDDING_DIMENSIONS", default_dims)),
        openai_api_key=openai_api_key,
        openai_base_url=openai_base_url,
        azure_openai_endpoint=azure_endpoint,
        azure_openai_api_key=azure_key,
        azure_openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        search_endpoint=search_endpoint,
        search_api_key=search_key,
        search_index=os.getenv("AZURE_SEARCH_INDEX", "docs-index"),
        local_index_path=os.getenv("LOCAL_INDEX_PATH", "data/local_index.json"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150")),
        top_k=int(os.getenv("TOP_K", "5")),
        storage_connection_string=os.getenv("AZURE_STORAGE_CONNECTION_STRING") or None,
        storage_container=os.getenv("AZURE_STORAGE_CONTAINER", "documents"),
    )
