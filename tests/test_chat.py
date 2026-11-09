from src.rag.chat import attribute_sources, cited_refs


CV_CHUNK = {
    "source": "RickBain.pdf", "page": 1, "ref": 1,
    "content": "Centroid-X-GAN - Medical Text-to-X-Ray Image Synthesis - ICDEC 2024 (Research Publication). "
               "Published at the 3rd International Conference on Data, Electronics & Computing (ICDEC 2024). "
               "Deployed on Hostinger VPS (Nginx, PM2), migrated to GCP with Dockerized CI/CD workflows.",
}
BIBLIOGRAPHY_CHUNK = {
    "source": "book.pdf", "page": 431, "ref": 2,
    "content": "BIBLIOGRAPHY PROCEEDINGS IJCAI-1 Walker, D. E., and Norton, L. M. (Eds.) 1969. International "
               "Joint Conference on Artificial Intelligence. Washington, D.C.; May.",
}


def test_attribution_picks_the_chunk_the_answer_came_from():
    question = "At which conference was Centroid-X-GAN published?"
    answer = ("The Centroid-X-GAN was published at the 3rd International Conference on Data, "
              "Electronics & Computing (ICDEC 2024).")
    assert attribute_sources(question, answer, [BIBLIOGRAPHY_CHUNK, CV_CHUNK]) == [CV_CHUNK]


def test_attribution_handles_two_questions_in_one_answer():
    question = "At which conference was Centroid-X-GAN published? How was the platform deployed?"
    answer = ("1. ICDEC 2024, the 3rd International Conference on Data, Electronics & Computing. "
              "2. Deployed on Hostinger VPS (Nginx, PM2) and migrated to GCP with Dockerized CI/CD workflows.")
    assert attribute_sources(question, answer, [BIBLIOGRAPHY_CHUNK, CV_CHUNK]) == [CV_CHUNK]


def test_attribution_falls_back_to_top_chunk():
    chunks = [CV_CHUNK, BIBLIOGRAPHY_CHUNK]
    assert attribute_sources("anything", "Totally unrelated zebra.", chunks) == [CV_CHUNK]
    assert attribute_sources("anything", "", chunks) == [CV_CHUNK]


def test_single_citation():
    assert cited_refs("It was published at ICDEC 2024 [1].", 5) == [1]


def test_multiple_and_duplicate_citations_keep_order():
    assert cited_refs("A [3]. B [1]. C [3].", 5) == [3, 1]


def test_adjacent_and_comma_citations():
    assert cited_refs("Both agree [1][2] and also [3, 4].", 5) == [1, 2, 3, 4]


def test_out_of_range_numbers_are_ignored():
    assert cited_refs("See [0], [6] and [2].", 5) == [2]


def test_no_citations():
    assert cited_refs("I couldn't find that in the documents.", 5) == []
    assert cited_refs("Version [2024] released.", 5) == []
