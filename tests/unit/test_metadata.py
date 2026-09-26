from datetime import date

from app.rag.metadata import extract_location, extract_published_date


def test_iso_date_from_header_hint() -> None:
    assert extract_published_date("2026-08-20T10:00:00Z", "no date here") == date(2026, 8, 20)


def test_long_form_dates() -> None:
    assert extract_published_date(None, "Announced on September 2, 2026 by the firm") == date(2026, 9, 2)
    assert extract_published_date(None, "said on 2 September 2026 that") == date(2026, 9, 2)


def test_impossible_date_returns_none() -> None:
    assert extract_published_date(None, "Published: 2026-02-31") is None


def test_only_scans_top_of_document() -> None:
    assert extract_published_date(None, ("filler " * 200) + "2026-01-01") is None


def test_location_most_frequent_known_place() -> None:
    assert extract_location("Whitefield office, Whitefield team, and one in Koramangala") == "Whitefield"


def test_unknown_location_is_none() -> None:
    assert extract_location("A company in Atlantis") is None