import pytest

from app.ingestion import normalizers as n


@pytest.mark.parametrize("raw,expected", [
    ("ABC Technologies Pvt Ltd", "abc"),
    ("ABC Tech", "abc"),
    ("abc-technologies", "abc"),
    ("Nova Labs Pvt. Ltd.", "nova labs"),
    ("Café Coffee Day Limited", "cafe coffee day"),
    ("3M India Ltd", "3m"),
    ("Technologies", "technologies"),
    ("Tata & Sons", "tata and sons"),
])
def test_normalize_company_name(raw: str, expected: str) -> None:
    assert n.normalize_company_name(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("www.Example.com/", "https://www.example.com"),
    ("example.com/path/", "https://example.com/path"),
    ("http://Example.COM", "https://example.com"),
    ("javascript:alert(1)", None),
    ("ftp://example.com", None),
    ("not a url", None),
    ("", None),
    (None, None),
])
def test_normalize_url(raw: object, expected: str | None) -> None:
    assert n.normalize_url(raw) == expected


def test_extract_domain_strips_www() -> None:
    assert n.extract_domain("https://www.novalabs.example/careers") == "novalabs.example"


def test_registrable_hint_handles_two_part_suffix() -> None:
    assert n.registrable_hint("careers.abc.co.in") == "abc.co.in"
    assert n.registrable_hint("careers.abc.com") == "abc.com"


@pytest.mark.parametrize("raw,expected", [
    ("https://in.linkedin.com/company/Nova-Labs/about", None),
    ("https://www.linkedin.com/company/Nova-Labs/", "https://www.linkedin.com/company/nova-labs"),
    ("linkedin.com/company/acme", "https://www.linkedin.com/company/acme"),
    ("https://www.linkedin.com/in/some-person", None),
    ("https://evil.com/company/acme", None),
])
def test_normalize_linkedin(raw: str, expected: str | None) -> None:
    assert n.normalize_linkedin_url(raw) == expected


def test_phone_valid_and_invalid() -> None:
    assert n.normalize_phone("+91 98450 12345") == "+919845012345"
    assert n.normalize_phone("098450 12345") == "+919845012345"
    assert n.normalize_phone("not-a-phone") is None
    assert n.normalize_phone("12345") is None


@pytest.mark.parametrize("raw,expected", [
    ("Bangalore", "Bengaluru"),
    ("BLR", "Bengaluru"),
    ("gurgaon", "Gurugram"),
    ("Nagpur", "Nagpur"),
])
def test_normalize_city(raw: str, expected: str) -> None:
    assert n.normalize_city(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("1,200", 1200),
    ("~50", 50),
    ("12.0", 12),
    ("12.5", None),
    ("-5", None),
    ("abc", None),
    ("", None),
])
def test_parse_int(raw: str, expected: int | None) -> None:
    assert n.parse_int(raw) == expected


def test_email() -> None:
    assert n.normalize_email(" Hello@Nova.EXAMPLE ") == "hello@nova.example"
    assert n.normalize_email("nope") is None