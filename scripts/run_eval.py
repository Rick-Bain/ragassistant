

import argparse
import csv
import dataclasses
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from src.rag.chat import answer_question
from src.rag.config import get_settings
from src.rag.embeddings import get_openai_client
from src.rag.evaluation import (
    breakdown,
    failure_reasons,
    is_refusal,
    keyword_recall,
    parse_verdict,
    source_hit,
    summarize,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QUESTIONS = ROOT / "eval" / "questions.json"
RESULTS_DIR = ROOT / "eval" / "results"

JUDGE_PROMPT = """You are grading an answer produced by a question-answering system.

Question: {question}
Reference answer: {expected}
Candidate answer: {answer}

Reply with exactly one word:
CORRECT   - the candidate contains the key facts of the reference answer
PARTIAL   - it has some of the key facts but misses or gets wrong something important
INCORRECT - it is wrong, unrelated, or says the information was not found"""


def pct(value) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def judge_answer(client, model: str, question: str, expected: str, answer: str) -> float | None:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": JUDGE_PROMPT.format(question=question, expected=expected, answer=answer),
            }
        ],
        temperature=0,
    )
    return parse_verdict(response.choices[0].message.content or "")


def evaluate_question(q: dict, settings, args, judge_client) -> dict:
    expected_source = q.get("expected_source")
    answerable = bool(expected_source)
    record = {
        "id": q["id"],
        "document": q.get("document", ""),
        "type": q.get("type", ""),
        "question": q["question"],
        "expected_answer": q["expected_answer"],
        "answerable": answerable,
    }

    start = time.perf_counter()
    try:
        result = answer_question(q["question"], settings)
    except Exception as exc:  # keep going; one bad question shouldn't kill the whole run
        record["error"] = f"{type(exc).__name__}: {exc}"
        return record
    record["latency"] = time.perf_counter() - start

    answer = result["answer"] or ""
    refused = is_refusal(answer)
    recall = keyword_recall(q["expected_answer"], answer) if answerable else 0.0

    record.update(
        answer=answer,
        refused=refused,
        cited=result["cited"],
        recall=recall,
        retrieval_hit=source_hit(expected_source, result["retrieved"]),
        source_hit=source_hit(expected_source, result["sources"]),
        retrieved=[(c["source"], c["page"]) for c in result["retrieved"]],
        shown=[(c["source"], c["page"]) for c in result["sources"]],
        correct=bool(answerable and not refused and recall >= args.threshold),
        judge_score=None,
    )
    if args.judge and answerable:
        try:
            record["judge_score"] = judge_answer(
                judge_client, args.judge_model or settings.chat_model,
                q["question"], q["expected_answer"], answer,
            )
        except Exception as exc:
            record["judge_error"] = f"{type(exc).__name__}: {exc}"
    return record


def print_report(label: str, settings, summary: dict, records: list[dict], args) -> None:
    line = "-" * 60
    print(f"\n{line}\nRESULTS: {label}\n{line}")
    print(f"Provider: {settings.provider} ({settings.chat_model}) | Store: {settings.vector_store}")
    print(f"Embeddings: {settings.embedding_model} | chunk size {settings.chunk_size} | top-k {settings.top_k}")
    print(
        f"Questions: {summary['n_questions']} "
        f"({summary['n_answerable']} answerable, {summary['n_unanswerable']} unanswerable, "
        f"{summary['n_errors']} errors)\n"
    )
    print(f"Retrieval hit rate ........ {pct(summary['retrieval_hit'])}   (right file in search results)")
    print(f"Right source shown ........ {pct(summary['source_hit'])}")
    print(f"Answer correct (words) .... {pct(summary['answer_correct'])}   (recall >= {args.threshold})")
    if args.judge:
        print(f"Answer correct (judge) .... {pct(summary['judge_score'])}")
    print(f"Refusal accuracy .......... {pct(summary['refusal_accuracy'])}   (says 'not found' when it should)")
    print(f"False refusal rate ........ {pct(summary['false_refusal'])}   (lower is better)")
    print(f"Model cites by itself ..... {pct(summary['citation_rate'])}")
    mean, p95 = summary["latency_mean"], summary["latency_p95"]
    if mean is not None:
        print(f"Latency ................... {mean:.1f}s mean, {p95:.1f}s p95")

    print("\nBy document (answerable questions)")
    for name, row in breakdown(records, "document").items():
        print(f"  {name:<14} n={row['n']:<3} retrieval {pct(row['retrieval_hit']):>7}   correct {pct(row['answer_correct']):>7}")

    failures = [(r, failure_reasons(r)) for r in records]
    failures = [(r, reasons) for r, reasons in failures if reasons]
    print(f"\nProblems found: {len(failures)}")
    for r, reasons in failures[:15]:
        print(f"  {r['id']} [{r['document']}] {r['question'][:70]}")
        for reason in reasons:
            print(f"      - {reason}")
    if len(failures) > 15:
        print(f"  ... and {len(failures) - 15} more (see the JSON file)")

    print("\nMarkdown row for your README results table:")
    print("| Configuration | Retrieval hit | Answer correct | Refusal accuracy | Latency (mean) |")
    print("|---|---|---|---|---|")
    correct = summary["judge_score"] if args.judge else summary["answer_correct"]
    lat = "n/a" if mean is None else f"{mean:.1f}s"
    print(f"| {label} | {pct(summary['retrieval_hit'])} | {pct(correct)} | {pct(summary['refusal_accuracy'])} | {lat} |")


def save_results(label: str, settings, summary: dict, records: list[dict], args) -> tuple[Path, Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_label = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in label)
    json_path = RESULTS_DIR / f"{safe_label}_{stamp}.json"
    config = {
        "label": label,
        "timestamp": stamp,
        "provider": settings.provider,
        "vector_store": settings.vector_store,
        "chat_model": settings.chat_model,
        "embedding_model": settings.embedding_model,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "top_k": settings.top_k,
        "word_recall_threshold": args.threshold,
        "judged": args.judge,
    }
    json_path.write_text(
        json.dumps({"config": config, "summary": summary, "records": records}, indent=2, default=str),
        encoding="utf-8",
    )

    csv_path = RESULTS_DIR / "summary.csv"
    row = {**config, **{k: v for k, v in summary.items()}}
    is_new = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)
    return json_path, csv_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate the RAG pipeline on a question set.")
    p.add_argument("--label", default="run", help="name for this experiment, e.g. 'chunk500-topk3'")
    p.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    p.add_argument("--top-k", type=int, help="override TOP_K for this run")
    p.add_argument("--threshold", type=float, default=0.6, help="word-recall needed to count as correct")
    p.add_argument("--judge", action="store_true", help="also grade answers with an LLM judge")
    p.add_argument("--judge-model", help="model for the judge (default: the chat model)")
    p.add_argument("--only", help="only questions whose 'document' equals this (cv, book, ...)")
    p.add_argument("--limit", type=int, help="only the first N questions")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()
    if args.top_k:
        settings = dataclasses.replace(settings, top_k=args.top_k)

    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    if args.only:
        questions = [q for q in questions if q.get("document") == args.only]
    if args.limit:
        questions = questions[: args.limit]
    if not questions:
        print("No questions to run.")
        return 1

    judge_client = get_openai_client(settings) if args.judge else None

    print(f"Running {len(questions)} questions ({settings.provider}/{settings.chat_model}, "
          f"store: {settings.vector_store}) ...")
    records, consecutive_errors = [], 0
    for i, q in enumerate(questions, start=1):
        record = evaluate_question(q, settings, args, judge_client)
        records.append(record)

        if record.get("error"):
            consecutive_errors += 1
            status = f"ERROR {record['error'][:60]}"
        else:
            consecutive_errors = 0
            problems = failure_reasons(record)
            status = ("ok" if not problems else "PROBLEM") + f"  {record['latency']:.1f}s"
        print(f"[{i}/{len(questions)}] {q['id']} {status}", flush=True)

        if consecutive_errors >= 3:
            print("\nStopping: 3 errors in a row. Is Ollama running and the index created and ingested?")
            print(f"Last error: {record['error']}")
            return 1

    summary = summarize(records)
    print_report(args.label, settings, summary, records, args)
    json_path, csv_path = save_results(args.label, settings, summary, records, args)
    print(f"\nSaved: {json_path}\nSummary row added to: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
