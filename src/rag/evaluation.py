"""Metrics for evaluating the RAG pipeline. Pure functions, no I/O, so they are easy to test.

What we measure, and why:

  retrieval_hit    Did the search return a chunk from the right file? If not, the model never had
                   a chance. This tells you whether to tune chunking / top-k / search.
  source_hit       Is the right file among the sources shown to the user?
  correct          Does the answer contain the key facts of the reference answer? (word recall,
                   or an LLM judge score if --judge is used)
  refusal          For questions that are NOT in the documents, did the system say so instead of
                   making something up? (the hallucination test)
  false_refusal    For questions that ARE answerable, did it wrongly say "I couldn't find that"?
  citation_rate    How often the model cites sources with [1], [2] by itself.
  latency          Seconds per question.
"""

import math
import re

REFUSAL_MARKERS = (
    "couldn't find",
    "could not find",
    "can't find",
    "cannot find",
    "not found in the",
    "does not contain",
    "doesn't contain",
    "do not contain",
    "no information",
    "not mentioned",
    "not provided",
)

_STOPWORDS = frozenset(
    """a an and are as at be been by can did do does for from had has have how in into is it its
    of on or that the their there these this those to was were what when where which who whom
    why will with would you your here also than then them they his her him she""".split()
)


def normalize_source(name: str) -> str:
    """Make file names comparable: 'My File ( v2 ).pdf' and 'My_File___v2__.pdf.txt' match."""
    name = name.lower().replace(".pdf", "").replace(".txt", "")
    return re.sub(r"[^a-z0-9]", "", name)


def source_hit(expected, chunks: list[dict]) -> bool | None:
    """True if every expected file appears among the chunks. None if the question has no source."""
    if not expected:
        return None
    wanted = [expected] if isinstance(expected, str) else list(expected)
    found = {normalize_source(c["source"]) for c in chunks}
    return all(normalize_source(w) in found for w in wanted)


def is_refusal(answer: str) -> bool:
    lowered = answer.lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def _tokens(text: str) -> set[str]:
    out = set()
    for token in re.findall(r"\d+(?:\.\d+)?|[a-z]+", text.lower()):
        if token[0].isdigit():
            out.add(token)
            continue
        if len(token) <= 2 or token in _STOPWORDS:
            continue
        if len(token) > 4 and token.endswith("s"):  # crude plural handling
            token = token[:-1]
        out.add(token)
    return out


def keyword_recall(expected: str, answer: str) -> float:
    """Fraction of the reference answer's meaningful words (and numbers) found in the answer."""
    wanted = _tokens(expected)
    if not wanted:
        return 0.0
    return len(wanted & _tokens(answer)) / len(wanted)


def parse_verdict(text: str) -> float | None:
    """Turn an LLM judge's reply into a score: CORRECT=1, PARTIAL=0.5, INCORRECT=0."""
    match = re.search(r"\b(INCORRECT|PARTIAL|CORRECT)\b", text.upper())
    if not match:
        return None
    return {"INCORRECT": 0.0, "PARTIAL": 0.5, "CORRECT": 1.0}[match.group(1)]


def _mean(values) -> float | None:
    values = [float(v) for v in values if v is not None]
    return sum(values) / len(values) if values else None


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(p / 100 * len(ordered)) - 1)
    return ordered[index]


def summarize(records: list[dict]) -> dict:
    """Aggregate per-question records into the headline numbers.

    Each record needs: answerable, retrieval_hit, source_hit, correct, judge_score,
    refused, cited, latency, and optionally error.
    """
    ok = [r for r in records if not r.get("error")]
    answerable = [r for r in ok if r["answerable"]]
    unanswerable = [r for r in ok if not r["answerable"]]
    answered = [r for r in answerable if not r["refused"]]

    return {
        "n_questions": len(records),
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "n_errors": len(records) - len(ok),
        "retrieval_hit": _mean(r["retrieval_hit"] for r in answerable),
        "source_hit": _mean(r["source_hit"] for r in answerable),
        "answer_correct": _mean(r["correct"] for r in answerable),
        "judge_score": _mean(r["judge_score"] for r in answerable),
        "false_refusal": _mean(r["refused"] for r in answerable),
        "refusal_accuracy": _mean(r["refused"] for r in unanswerable),
        "citation_rate": _mean(r["cited"] for r in answered),
        "latency_mean": _mean(r["latency"] for r in ok),
        "latency_p95": percentile([r["latency"] for r in ok], 95),
    }


def breakdown(records: list[dict], key: str) -> dict[str, dict]:
    """Retrieval hit rate and correctness per document (or per question type)."""
    groups: dict[str, list[dict]] = {}
    for r in records:
        if r["answerable"] and not r.get("error"):
            groups.setdefault(r[key], []).append(r)
    return {
        name: {
            "n": len(items),
            "retrieval_hit": _mean(i["retrieval_hit"] for i in items),
            "answer_correct": _mean(i["correct"] for i in items),
        }
        for name, items in groups.items()
    }


def failure_reasons(record: dict) -> list[str]:
    """Plain-language reasons a question went wrong (empty list if it went fine)."""
    if record.get("error"):
        return [f"error: {record['error']}"]
    reasons = []
    if record["answerable"]:
        if record["retrieval_hit"] is False:
            reasons.append("retrieval miss (right file not in results)")
        elif record["source_hit"] is False:
            reasons.append("right file retrieved but not shown as a source")
        if record["refused"]:
            reasons.append("false refusal (said it couldn't find it)")
        elif not record["correct"]:
            reasons.append(f"wrong or incomplete answer (word recall {record['recall']:.2f})")
    elif not record["refused"]:
        reasons.append("hallucination (should have said it couldn't find it)")
    return reasons
