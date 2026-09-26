from pathlib import Path

import pytest

from app.rag.loaders import DocumentLoadError, load_document


def test_txt_loaded(tmp_path: Path) -> None:
    p = tmp_path / "note.txt"
    p.write_text("Hello world", encoding="utf-8")
    docs = load_document(p)
    assert docs[0].title == "note" and docs[0].text == "Hello world"


def test_html_scripts_and_nav_removed_metadata_kept(tmp_path: Path) -> None:
    p = tmp_path / "a.html"
    p.write_text(
        '<html><head><title>T</title><meta property="article:published_time" content="2026-08-20">'
        "<script>alert(1)</script></head><body><nav>menu</nav><p>Real body text.</p>"
        "<footer>foot</footer></body></html>",
        encoding="utf-8",
    )
    doc = load_document(p)[0]
    assert "alert" not in doc.text and "menu" not in doc.text and "Real body text." in doc.text
    assert doc.meta["published_raw"] == "2026-08-20" and doc.title == "T"


def test_csv_rows_become_documents(tmp_path: Path) -> None:
    p = tmp_path / "d.csv"
    p.write_text("name,city\nAcme,Pune\nBeta,Delhi\n", encoding="utf-8")
    docs = load_document(p)
    assert len(docs) == 2 and "name: Acme" in docs[0].text


def test_unsupported_and_missing_files_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "x.exe"
    bad.write_bytes(b"MZ")
    with pytest.raises(DocumentLoadError, match="unsupported"):
        load_document(bad)
    with pytest.raises(DocumentLoadError, match="not found"):
        load_document(tmp_path / "nope.txt")


def test_malformed_pdf_rejected_not_crashed(tmp_path: Path) -> None:
    p = tmp_path / "broken.pdf"
    p.write_bytes(b"%PDF-1.4 this is not really a pdf")
    with pytest.raises(DocumentLoadError):
        load_document(p)


def test_empty_file_rejected(tmp_path: Path) -> None:
    p = tmp_path / "empty.txt"
    p.write_text("   \n  ", encoding="utf-8")
    with pytest.raises(DocumentLoadError, match="no text"):
        load_document(p)