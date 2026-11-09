

from src.rag.config import get_settings
from src.rag.vector_store import get_store


def main() -> None:
    settings = get_settings()
    print(f"Provider: {settings.provider} | Vector store: {settings.vector_store}")
    print(get_store(settings).create_index())


if __name__ == "__main__":
    main()
