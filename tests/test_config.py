import pytest

from src.rag import config

ALL_VARS = [
    "PROVIDER", "VECTOR_STORE", "OPENAI_API_KEY", "OPENAI_BASE_URL", "EMBEDDING_DIMENSIONS",
    "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_CHAT_DEPLOYMENT",
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "AZURE_SEARCH_ENDPOINT", "AZURE_SEARCH_API_KEY",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ALL_VARS:
        monkeypatch.delenv(name, raising=False)


def test_defaults_are_free_local_stack():
    s = config.get_settings()
    assert (s.provider, s.vector_store) == ("ollama", "local")
    assert s.embedding_dimensions == 768
    assert "11434" in s.openai_base_url


def test_azure_everything(monkeypatch):
    for k, v in {
        "PROVIDER": "azure", "VECTOR_STORE": "azure_search",
        "AZURE_OPENAI_ENDPOINT": "https://r.openai.azure.com/", "AZURE_OPENAI_API_KEY": "k",
        "AZURE_OPENAI_CHAT_DEPLOYMENT": "chat1", "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": "emb1",
        "AZURE_SEARCH_ENDPOINT": "https://x.search.windows.net", "AZURE_SEARCH_API_KEY": "k",
    }.items():
        monkeypatch.setenv(k, v)
    s = config.get_settings()
    assert (s.chat_model, s.embedding_model, s.embedding_dimensions) == ("chat1", "emb1", 1536)


def test_mixed_ollama_models_with_azure_search(monkeypatch):
    monkeypatch.setenv("VECTOR_STORE", "azure_search")
    monkeypatch.setenv("AZURE_SEARCH_ENDPOINT", "https://x.search.windows.net")
    monkeypatch.setenv("AZURE_SEARCH_API_KEY", "k")
    s = config.get_settings()
    assert (s.provider, s.vector_store) == ("ollama", "azure_search")


def test_missing_values_give_clear_errors(monkeypatch):
    monkeypatch.setenv("PROVIDER", "openai")
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        config.get_settings()
    monkeypatch.setenv("PROVIDER", "ollama")
    monkeypatch.setenv("VECTOR_STORE", "azure_search")
    with pytest.raises(RuntimeError, match="AZURE_SEARCH_ENDPOINT"):
        config.get_settings()


def test_bad_values_rejected(monkeypatch):
    monkeypatch.setenv("PROVIDER", "banana")
    with pytest.raises(RuntimeError, match="PROVIDER"):
        config.get_settings()
    monkeypatch.setenv("PROVIDER", "ollama")
    monkeypatch.setenv("VECTOR_STORE", "nope")
    with pytest.raises(RuntimeError, match="VECTOR_STORE"):
        config.get_settings()
