# Document Assistant (RAG): runs locally or on Azure

A chat app that answers questions about your PDF documents and **cites the file and page** for every answer. It uses hybrid (keyword + vector) retrieval and refuses to guess when the documents don't contain the answer.

The same code runs two ways, selected by two settings in `.env`:

| Mode | `PROVIDER` | `VECTOR_STORE` | Cost | Use it for |
|---|---|---|---|---|
| **Local** | `ollama` | `local` | Free, offline | Development, testing, evaluation |
| **Azure** | `azure` | `azure_search` | Azure credit | Live demo and deployment |

(`PROVIDER=openai` is also supported, and the two settings can be mixed, e.g. Ollama models with Azure AI Search.)

> **Status:** Stage 1 (core RAG). Stage 2 adds cloud deployment, Key Vault, evaluation and monitoring.

<!-- TODO: add a screenshot or demo GIF here -->
<!-- TODO: add link to live demo and 2-3 min demo video -->

## Features
- PDF ingestion with overlapping, page-aware chunking
- Hybrid search: BM25 keyword + vector similarity, fused with reciprocal rank fusion
- Grounded answers that say "I couldn't find that" instead of guessing
- Source citations with file name and page number
- Provider and vector-store abstraction: swap Ollama / OpenAI / Azure OpenAI and a local index / Azure AI Search without code changes
- Safeguard against mixing embeddings from different models
- Unit tests for chunking, retrieval and configuration

## Tech stack
| Layer | Local mode | Azure mode |
|---|---|---|
| LLM + embeddings | Ollama (`llama3.2:3b`, `nomic-embed-text`) | Azure OpenAI |
| Retrieval | Local JSON index (NumPy cosine + BM25) | Azure AI Search (HNSW vector + BM25) |
| Raw document storage | Local folder | Azure Blob Storage (optional) |
| App | Python, Streamlit | Python, Streamlit |

See [docs/architecture.md](docs/architecture.md) for the diagram and design decisions.

## Project structure
```
├── app/streamlit_app.py      # chat UI (shows which provider/store is active)
├── src/rag/
│   ├── config.py             # settings + the two switches
│   ├── loader.py             # PDF reading + chunking
│   ├── embeddings.py         # client for Ollama / OpenAI / Azure OpenAI
│   ├── vector_store.py       # LocalStore and AzureSearchStore (same interface)
│   └── chat.py               # retrieve + grounded answer
├── scripts/
│   ├── create_index.py       # one-time index setup / reset
│   └── ingest.py             # (blob upload) + chunk + embed + index
├── tests/                    # pytest: chunking, local store, config
├── data/sample_docs/         # put your PDFs here (git-ignored)
├── eval/questions.json       # test questions for Stage 2
├── docs/architecture.md
├── .env.example
├── requirements.txt
└── requirements-dev.txt
```

## Quick start: free local mode

1. Install [Ollama](https://ollama.com) and pull two models:
   ```bash
   ollama pull llama3.2:3b
   ollama pull nomic-embed-text
   ```
2. Set up Python:
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   cp .env.example .env             # defaults already select the local mode
   ```
3. Add a few PDFs to `data/sample_docs/`, then:
   ```bash
   python -m scripts.create_index
   python -m scripts.ingest
   streamlit run app/streamlit_app.py
   ```
4. Run the tests: `pip install -r requirements-dev.txt && pytest`

## Switching to Azure

1. Create the resources: Azure AI Search (Free tier is enough: 50 MB, 3 indexes), Azure OpenAI with a chat and an embedding deployment, and optionally a Storage account.
2. In `.env`, set:
   ```
   PROVIDER=azure
   VECTOR_STORE=azure_search
   EMBEDDING_DIMENSIONS=1536
   AZURE_SEARCH_INDEX=docs-index-azure     # new name: vector size is fixed per index
   ```
   and fill in the `AZURE_*` values (the commented block in `.env.example`).
3. Rebuild and re-ingest, because embeddings from different models can't be mixed:
   ```bash
   python -m scripts.create_index
   python -m scripts.ingest
   streamlit run app/streamlit_app.py
   ```

The sidebar of the app shows which models and vector store are active.

## Roadmap
- [x] Stage 1: ingestion, hybrid search, grounded chat with citations, local and Azure modes
- [ ] Stage 2: deploy to Azure, Key Vault + Managed Identity, evaluation set, Application Insights
- [ ] Stage 3: one agent tool (e.g. summarize document or extract invoice fields)

## Results
_Fill in after Stage 2: evaluation table comparing chunk sizes, and local vs Azure configurations on the same question set._

| Configuration | Groundedness | Retrieval hit rate |
|---|---|---|
| Ollama + local, 500 chars | TBD | TBD |
| Ollama + local, 1000 chars | TBD | TBD |
| Azure OpenAI + AI Search, 1000 chars | TBD | TBD |

## What I learned
_Write 4-5 honest sentences: what was hard, what surprised you, what you would change. Interviewers read this._
