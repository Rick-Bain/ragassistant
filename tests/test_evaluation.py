from src.rag.evaluation import (
    breakdown,
    failure_reasons,
    is_refusal,
    keyword_recall,
    normalize_source,
    parse_verdict,
    percentile,
    source_hit,
    summarize,
)


def test_normalize_source_matches_different_spellings_of_one_file():
    a = "Principles of Artificial Intelligence ( PDFDrive ).pdf"
    b = "Principles_of_Artificial_Intelligence___PDFDrive__.pdf.txt"
    assert normalize_source(a) == normalize_source(b)
    assert normalize_source("RickBain.pdf") != normalize_source("MEDICAL_IMAGE.pdf")


def test_source_hit_single_list_and_none():
    chunks = [{"source": "RickBain.pdf"}, {"source": "MEDICAL_IMAGE.pdf"}]
    assert source_hit("RickBain.pdf", chunks) is True
    assert source_hit("other.pdf", chunks) is False
    assert source_hit(["RickBain.pdf", "MEDICAL_IMAGE.pdf"], chunks) is True
    assert source_hit(["RickBain.pdf", "other.pdf"], chunks) is False  # all must be present
    assert source_hit(None, chunks) is None


def test_refusal_detection():
    assert is_refusal("I couldn't find that in the documents.")
    assert is_refusal("The documents do not contain this information.")
    assert not is_refusal("It was published at ICDEC 2024.")


def test_keyword_recall_rewards_matching_facts():
    expected = "Nils J. Nilsson, Stanford University."
    assert keyword_recall(expected, "It was written by Nils Nilsson of Stanford University.") == 1.0
    assert keyword_recall(expected, "It was written by someone else.") < 0.5


def test_keyword_recall_handles_numbers_and_plurals():
    expected = "FID scores as low as 1.11 for key disease classes."
    # 6 of the 7 key words match (the answer never says "low"), so recall is about 0.86
    assert 0.8 <= keyword_recall(expected, "The best FID score is 1.11 for the key disease classes.") < 1.0
    assert keyword_recall(expected, "The best FID score is 2.5.") < 0.6


def test_parse_verdict():
    assert parse_verdict("CORRECT") == 1.0
    assert parse_verdict("partial - misses the date") == 0.5
    assert parse_verdict("INCORRECT") == 0.0  # must not be read as CORRECT
    assert parse_verdict("I am not sure") is None


def test_percentile():
    assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95) == 10
    assert percentile([], 95) is None


def make(**overrides):
    base = dict(
        id="q", document="cv", type="factual", answerable=True, retrieval_hit=True, source_hit=True,
        correct=True, judge_score=None, refused=False, cited=True, latency=1.0, recall=1.0,
    )
    base.update(overrides)
    return base


def test_summarize_headline_numbers():
    records = [
        make(),
        make(retrieval_hit=False, source_hit=False, correct=False, cited=False),
        make(answerable=False, retrieval_hit=None, source_hit=None, correct=False, refused=True),
        make(answerable=False, retrieval_hit=None, source_hit=None, correct=False, refused=False),
        make(error="boom"),
    ]
    s = summarize(records)
    assert (s["n_questions"], s["n_answerable"], s["n_unanswerable"], s["n_errors"]) == (5, 2, 2, 1)
    assert s["retrieval_hit"] == 0.5
    assert s["answer_correct"] == 0.5
    assert s["refusal_accuracy"] == 0.5
    assert s["false_refusal"] == 0.0
    assert s["citation_rate"] == 0.5


def test_breakdown_groups_by_document():
    records = [make(document="cv"), make(document="cv", correct=False), make(document="book")]
    result = breakdown(records, "document")
    assert result["cv"]["n"] == 2 and result["cv"]["answer_correct"] == 0.5
    assert result["book"]["n"] == 1


def test_failure_reasons():
    assert failure_reasons(make()) == []
    assert "retrieval miss" in failure_reasons(make(retrieval_hit=False, source_hit=False, correct=False))[0]
    assert "false refusal" in " ".join(failure_reasons(make(refused=True, correct=False)))
    assert "hallucination" in failure_reasons(make(answerable=False, refused=False, correct=False))[0]
    assert failure_reasons(make(answerable=False, refused=True, correct=False)) == []
