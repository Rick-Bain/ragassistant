"""The RAG step: retrieve relevant chunks, then ask the model to answer using only those."""

import re

from .config import Settings, get_settings
from .embeddings import embed_texts, get_openai_client
from .vector_store import get_store

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the provided sources.

Rules:
- If the sources do not contain the answer, say "I couldn't find that in the documents." Do not guess.
- After every claim, cite the source number in square brackets, like [1] or [2].
  Example: "The leave allowance is 20 days [2]."
- If the user asks several questions, answer each one and cite its sources.
- Keep answers clear and concise."""

CITATION_REMINDER = (
    "Remember: put the source number in square brackets after each claim, for example [1]."
)

REFUSAL_PHRASE = "couldn't find that in the documents"

_STOPWORDS = frozenset(
    """a an and are as at be been by can did do does for from had has have how in into is it its
    of on or that the their there these this those to was were what when where which who whom
    why will with would you your here answer answers question questions according""".split()
)


def _build_context(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"[{c['ref']}] (file: {c['source']}, page {c['page']})\n{c['content']}" for c in chunks
    )


def cited_refs(answer: str, n_chunks: int) -> list[int]:
    """Source numbers the answer actually cites, in order. Handles [1], [1][2] and [1, 2]."""
    refs: list[int] = []
    for group in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", answer):
        for number in re.findall(r"\d+", group):
            n = int(number)
            if 1 <= n <= n_chunks and n not in refs:
                refs.append(n)
    return refs


def _content_tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2 and t not in _STOPWORDS}


def attribute_sources(question: str, answer: str, chunks: list[dict], max_sources: int = 3) -> list[dict]:
    """Fallback for models that don't cite: pick the chunks the answer most likely came from.

    Words that were already in the question are ignored (they appear in the answer just because
    the model echoed the question). The remaining words come from the sources, so each chunk is
    scored by how many of them it contains. Keeps chunks scoring at least half of the best one.
    """
    new_words = _content_tokens(answer) - _content_tokens(question)
    if not new_words:
        return chunks[:1]

    scored = [(len(new_words & _content_tokens(c["content"])) / len(new_words), c) for c in chunks]
    best = max(score for score, _ in scored)
    if best == 0:
        return chunks[:1]

    kept = [(score, c) for score, c in scored if score >= 0.5 * best]
    kept.sort(key=lambda pair: -pair[0])
    return [c for _, c in kept[:max_sources]]


def answer_question(question: str, settings: Settings | None = None) -> dict:
    """Answer a question.

    Returns:
      answer    - the model's reply
      sources   - the chunks the answer came from (what the UI shows)
      retrieved - every chunk that was retrieved (useful for debugging and evaluation)
      cited     - True if the sources come from the model's own [n] citations;
                  False if the model didn't cite and the sources were inferred from the text
    """
    settings = settings or get_settings()
    client = get_openai_client(settings)
    store = get_store(settings)

    query_vector = embed_texts(client, settings, [question])[0]
    chunks = store.search(question, query_vector)

    if not chunks:
        return {
            "answer": "I couldn't find that in the documents.",
            "sources": [],
            "retrieved": [],
            "cited": True,
        }

    for i, chunk in enumerate(chunks, start=1):
        chunk["ref"] = i

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Sources:\n{_build_context(chunks)}\n\n"
                f"Question: {question}\n\n{CITATION_REMINDER}"
            ),
        },
    ]
    response = client.chat.completions.create(
        model=settings.chat_model,  # model name (Ollama/OpenAI) or deployment name (Azure)
        messages=messages,
        temperature=0.1,
    )
    answer = response.choices[0].message.content

    refs = cited_refs(answer, len(chunks))
    if refs:
        sources, cited = [chunks[i - 1] for i in refs], True
    elif REFUSAL_PHRASE in answer.lower():
        sources, cited = [], True  # the model correctly said the answer isn't in the documents
    else:
        sources, cited = attribute_sources(question, answer, chunks), False

    return {"answer": answer, "sources": sources, "retrieved": chunks, "cited": cited}
