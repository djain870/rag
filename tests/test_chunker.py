from src.chunker_unused import chunk_text


def test_chunk_text():
    text = "a" * 2500

    chunks = chunk_text(
        text,
        chunk_size=1000,
        chunk_overlap=200
    )
    assert len(chunks) > 1
    assert all(len(chunk) <= 1000 for chunk in chunks)
