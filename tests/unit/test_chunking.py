import pytest

from app.rag.chunking import approx_tokens, chunk_text


def _para(n_words: int, tag: str) -> str:
    return " ".join(f"{tag}{i}" for i in range(n_words)) + "."


def test_short_text_is_single_chunk() -> None:
    chunks = chunk_text("One short paragraph.", 450, 60)
    assert len(chunks) == 1 and chunks[0].index == 0


def test_chunks_respect_target_size() -> None:
    text = "\n\n".join(_para(80, f"p{i}_") for i in range(20))
    chunks = chunk_text(text, target_tokens=200, overlap_tokens=30)
    assert len(chunks) > 1
    assert all(c.token_count <= 200 + 10 for c in chunks)


def test_overlap_carries_context_between_chunks() -> None:
    text = "\n\n".join(_para(80, f"p{i}_") for i in range(10))
    chunks = chunk_text(text, target_tokens=200, overlap_tokens=40)
    assert len(chunks) >= 2
    tail_words = set(chunks[0].content.split()[-5:])
    assert tail_words & set(chunks[1].content.split())


def test_no_content_lost() -> None:
    paras = [_para(60, f"q{i}_") for i in range(12)]
    chunks = chunk_text("\n\n".join(paras), target_tokens=150, overlap_tokens=20)
    joined = " ".join(c.content for c in chunks)
    for i in range(12):
        assert f"q{i}_0" in joined and f"q{i}_59" in joined


def test_oversized_unbroken_string_is_hard_split() -> None:
    chunks = chunk_text("x" * 5000, target_tokens=100, overlap_tokens=10)
    assert len(chunks) > 1
    assert all(c.token_count <= 110 for c in chunks)


def test_identical_text_same_hash_and_invalid_overlap_rejected() -> None:
    a, b = chunk_text("Same text.", 100, 10), chunk_text("Same text.", 100, 10)
    assert a[0].content_hash == b[0].content_hash
    with pytest.raises(ValueError):
        chunk_text("x", 100, 100)


def test_approx_tokens_never_zero() -> None:
    assert approx_tokens("") == 1