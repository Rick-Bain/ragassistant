# Evaluation

`questions.json` holds 44 questions with reference answers, written from the sample documents:
answerable questions (CV, certificate, minor-project paper, the AI textbook), two questions that
need two documents, and five that are **not** in the documents (to catch hallucination).

## Run it

```bash
python -m scripts.run_eval --label baseline
```

Useful options:

| Option | What it does |
|---|---|
| `--label NAME` | Name of this experiment (shows up in the results table) |
| `--top-k 3` | Override how many chunks are retrieved |
| `--judge` | Also grade answers with an LLM (see the note below) |
| `--judge-model NAME` | Use a different model as the judge |
| `--only cv` | Only one document group: `cv`, `certificate`, `minor_project`, `book`, `cross`, `unanswerable` |
| `--limit 10` | Only the first 10 questions (quick check) |
| `--threshold 0.6` | Word-recall needed to count an answer as correct |

Every run prints a report, saves details to `eval/results/<label>_<time>.json`, and appends one row to
`eval/results/summary.csv`. The report ends with a Markdown row you can paste into the README.

## What the numbers mean

| Metric | Meaning | If it's low |
|---|---|---|
| Retrieval hit rate | The right file is among the retrieved chunks | Tune chunk size, top-k, or search |
| Right source shown | The right file is among the sources shown to the user | Citation behavior of the model |
| Answer correct (words) | The answer contains the key words and numbers of the reference answer | Model or prompt problem (if retrieval is fine) |
| Answer correct (judge) | An LLM grades the answer: correct, partial or incorrect | Same as above |
| Refusal accuracy | On the 5 unanswerable questions, it says "I couldn't find that" | The model hallucinates |
| False refusal rate | It refuses a question that *is* answerable | Retrieval missed, or the model is too cautious |
| Model cites by itself | The model wrote `[1]`-style citations without the fallback | Prompt or model choice |

Read them together: **retrieval hit high but answer correct low** means the model is the weak part;
**retrieval hit low** means fix search before touching the model.

### Limits of the scoring
- Word recall is cheap and repeatable but crude: a correct answer phrased differently can score low,
  and a wrong answer that reuses the right words can score high. Check the "Problems found" list by eye.
- A small local model is an unreliable judge. When you have Azure OpenAI (or another strong model),
  use it with `--judge` for more trustworthy grading. Mention in your write-up which judge you used.
- 44 questions is enough to see big differences (e.g. 60% vs 85%), not small ones (e.g. 82% vs 84%).

## Running experiments properly

Chunk size is applied when you **ingest**, so each chunking experiment needs its own index.
Use a separate `LOCAL_INDEX_PATH` per experiment, and use the same settings for ingest and eval.

Linux / macOS:
```bash
CHUNK_SIZE=500 LOCAL_INDEX_PATH=data/idx_500.json python -m scripts.create_index
CHUNK_SIZE=500 LOCAL_INDEX_PATH=data/idx_500.json python -m scripts.ingest
CHUNK_SIZE=500 LOCAL_INDEX_PATH=data/idx_500.json python -m scripts.run_eval --label chunk500
```

Windows PowerShell:
```powershell
$env:CHUNK_SIZE=500; $env:LOCAL_INDEX_PATH="data/idx_500.json"
python -m scripts.create_index
python -m scripts.ingest
python -m scripts.run_eval --label chunk500
```

Suggested experiments (change one thing at a time):
1. Chunk size: 500, 1000, 1500
2. `TOP_K`: 3 vs 5 (no re-ingest needed, just `--top-k`)
3. Chat model: `llama3.2:3b` vs LFM3 (set `OLLAMA_CHAT_MODEL`; no re-ingest needed)
4. Azure: `PROVIDER=azure`, `VECTOR_STORE=azure_search` with a new `AZURE_SEARCH_INDEX`

For Azure, re-ingest into a new index and keep the same question file so the comparison is fair.

## Note on file names
`expected_source` is matched loosely (case, spaces, underscores and `.pdf`/`.txt` are ignored), so
`Principles of Artificial Intelligence ( PDFDrive ).pdf` matches the name used in `questions.json`.
