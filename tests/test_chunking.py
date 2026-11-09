from src.rag.loader import chunk_text


def test_short_text_is_one_chunk():
    assert chunk_text("Hello world.", 1000, 150) == ["Hello world."]


def test_empty_text_gives_no_chunks():
    assert chunk_text("   ", 1000, 150) == []


def test_long_text_is_split_within_limit():
    text = "This is a sentence about policy. " * 120
    chunks = chunk_text(text, 1000, 150)
    assert len(chunks) > 1
    assert all(len(c) <= 1000 for c in chunks)


def test_chunks_overlap():
    text = " ".join(f"word{i}" for i in range(600))
    chunks = chunk_text(text, 500, 100)
    # the end of one chunk should reappear at the start of the next
    tail = chunks[0].split()[-3:]
    assert all(word in chunks[1] for word in tail)
