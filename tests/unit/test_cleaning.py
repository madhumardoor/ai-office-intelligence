from app.rag.cleaning import clean_text, content_hash


def test_removes_control_chars_and_collapses_blank_lines() -> None:
    assert clean_text("a\x00b\n\n\n\nc") == "ab\n\nc"


def test_joins_pdf_hyphenation() -> None:
    assert clean_text("expan-\nsion plans") == "expansion plans"


def test_drops_page_number_lines() -> None:
    assert clean_text("Real text\n12\nPage 3 of 10\nMore text") == "Real text\nMore text"


def test_nfkc_normalisation() -> None:
    assert clean_text("\ufb01nance") == "finance"


def test_hash_ignores_whitespace_but_not_case() -> None:
    assert content_hash("a  b\n c") == content_hash("a b c")
    assert content_hash("Apple") != content_hash("apple")