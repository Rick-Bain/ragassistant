"""Create embeddings with Ollama, OpenAI or Azure OpenAI (same interface, different client)."""

from openai import AzureOpenAI, OpenAI

from .config import Settings


def get_openai_client(settings: Settings) -> OpenAI:
    """Return the right client for the configured provider.

    Ollama and OpenAI both use the plain OpenAI client (Ollama just has a different base_url).
    """
    if settings.provider == "azure":
        return AzureOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
        )
    return OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)


def embed_texts(client: OpenAI, settings: Settings, texts: list[str]) -> list[list[float]]:
    """Embed many texts, in batches to stay under request limits."""
    vectors: list[list[float]] = []
    batch_size = 16
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(model=settings.embedding_model, input=batch)
        vectors.extend(item.embedding for item in response.data)
    return vectors
