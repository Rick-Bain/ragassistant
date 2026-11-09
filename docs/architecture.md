# Architecture (Stage 1)

```
                INGESTION (run once per document set)
 PDFs ──► (Blob Storage: raw copy, Azure mode)
   │
   └──► pypdf text extraction ──► chunking (size + overlap, page-aware)
                                      │
                                      ▼
                    Embedding model  ◄── PROVIDER: ollama | openai | azure
                                      │
                                      ▼
                    Vector store     ◄── VECTOR_STORE: local | azure_search
              (text + vector + source/page metadata)

                QUERY (every question)
 Question ──► embed ──► hybrid search (BM25 + vector, rank-fused) ──► top-k chunks
                                                                          │
                                                                          ▼
                            Chat model with grounded prompt (same PROVIDER)
                                                                          │
                                                                          ▼
                                   Answer with [1] [2] citations + sources
```

## Why two switches
`PROVIDER` and `VECTOR_STORE` are independent. The app code only talks to two small
interfaces (an OpenAI-style client and a store with `create_index / upload_chunks / search`),
so development runs free and offline, and the Azure deployment is a configuration change.
Ollama and OpenAI share one client class because Ollama exposes an OpenAI-compatible API.

## Design decisions (fill these in as you build, interviewers love this)

| Decision | Choice | Why |
|---|---|---|
| Chunk size / overlap | 1000 / 150 chars | _Your reasoning. Update after Stage 2 experiments._ |
| Retrieval | Hybrid (BM25 + vector, reciprocal rank fusion) | _Why hybrid beat vector-only on your test questions._ |
| Local models | llama3.2:3b + nomic-embed-text | _Small enough for a laptop; note quality differences vs Azure OpenAI._ |
| Azure models | gpt-4o-mini class + text-embedding-3-small | _Cost, latency, quality._ |
| Prompt | Answer only from sources, cite numbers | _How this reduces hallucination._ |
| Embedding mismatch guard | Local index records model + dimensions | _Prevents silently mixing incompatible vectors._ |

## Known limitations (Stage 1)
- Text-only PDFs. Scanned PDFs need Azure AI Document Intelligence (planned).
- The local store reads the whole index per query, which is fine for a few thousand chunks, not for large corpora.
- Small local models follow citation instructions less reliably than hosted models; compare them in Stage 2.
- No user access control; everyone sees every document.
- Keys live in `.env`; Stage 2 moves them to Key Vault + Managed Identity.
- No automated evaluation yet; Stage 2 adds it.
