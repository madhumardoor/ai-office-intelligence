"""Pure normalization functions (no I/O) so they are trivially unit-testable."""

from __future__ import annotations

import re
import unicodedata
from urllib.parse import urlparse

import phonenumbers


# ---------------------------------------------------------------------------
# Legal-entity suffixes stripped for matching only.
# Display names remain unchanged.
# ---------------------------------------------------------------------------

_LEGAL_SUFFIXES = (
    r"private limited",
    r"pvt\.?\s*ltd\.?",
    r"pvt",
    r"limited",
    r"ltd\.?",
    r"llp",
    r"inc\.?",
    r"incorporated",
    r"corp\.?",
    r"corporation",
    r"llc",
    r"gmbh",
    r"co\.?",
    r"company",
    r"technologies",
    r"technology",
    r"tech",
    r"solutions",
    r"systems",
    r"software",
    r"india",
)

_SUFFIX_RE = re.compile(
    r"\b(?:" + "|".join(_LEGAL_SUFFIXES) + r")\b",
    re.IGNORECASE,
)

_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")
_WS = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Multi-part public suffixes common for Indian companies.
# ---------------------------------------------------------------------------

_TWO_PART_SUFFIXES = {
    "co.in",
    "org.in",
    "net.in",
    "gov.in",
    "ac.in",
    "co.uk",
    "com.au",
}


# ---------------------------------------------------------------------------
# City normalization
# ---------------------------------------------------------------------------

_CITY_ALIASES = {
    "bangalore": "Bengaluru",
    "bengaluru": "Bengaluru",
    "bengalooru": "Bengaluru",
    "blr": "Bengaluru",
    "bombay": "Mumbai",
    "mumbai": "Mumbai",
    "madras": "Chennai",
    "chennai": "Chennai",
    "calcutta": "Kolkata",
    "kolkata": "Kolkata",
    "gurgaon": "Gurugram",
    "gurugram": "Gurugram",
    "delhi": "Delhi",
    "new delhi": "Delhi",
    "hyderabad": "Hyderabad",
    "pune": "Pune",
}


# ---------------------------------------------------------------------------
# Known coworking operators
#
# These are used only for coworking-master listings where the source file
# does not contain an explicit coworking_operator column.
# ---------------------------------------------------------------------------

_COWORKING_OPERATORS: tuple[tuple[str, str], ...] = (
    # Existing operators
    ("91springboard", "91springboard"),
    ("awfis", "Awfis"),
    ("bluedesk", "BlueDesk"),
    ("hive spaces", "Hive Spaces"),
    ("hive space", "Hive Spaces"),
    ("indiqube", "IndiQube"),
    ("nimbus cowork", "Nimbus Cowork"),
    ("nimbus coworking", "Nimbus Cowork"),
    ("orbitwork", "OrbitWork"),
    ("wework", "WeWork"),

    # Operators identified from all_cities_cleaned.csv
    ("redbrick offices", "Redbrick Offices"),
    ("bhive workspace", "BHIVE Workspace"),
    ("namma office", "Namma Office"),
    ("attic space", "Attic Space"),
    ("unispace", "Unispace"),
    ("smartworks", "Smartworks"),
    ("315work avenue", "315Work Avenue"),
    ("anthill iq workspace", "Anthill IQ Workspace"),
    ("flex coworks", "Flex Coworks"),
    ("gopalan workspace", "Gopalan Workspace"),
    ("the work address", "The Work Address"),
    ("symbyont smart spaces", "Symbyont Smart Spaces"),
    ("quickstart spaces", "Quickstart Spaces"),
    ("teloz spaces", "Teloz Spaces"),
    ("goodworks cowork", "GoodWorks Cowork"),
    ("goodworks coworks", "GoodWorks Cowork"),
    ("office republic", "Office Republic"),
    ("aspire coworks", "Aspire Coworks"),
    ("the office pass", "The Office Pass"),
    ("hubstairs coworks", "Hubstairs CoWorks"),
    ("hubstairs cowork", "Hubstairs CoWorks"),
    ("the workvilla", "The Workvilla"),
    ("workden", "WorkDen"),
    ("urbandesk", "UrbanDesk"),
)

def detect_coworking_operator(value: object) -> str | None:
    """
    Detect a known coworking operator from a listing/company name.

    Examples:
        WeWork Galaxy -> WeWork
        WeWork Manyata Redwood -> WeWork
        Awfis Bannerghatta -> Awfis
        IndiQube Lexington -> IndiQube

    Returns None when no known operator is detected.
    """
    text = clean_text(value)

    if not text:
        return None

    lowered = text.casefold()

    for needle, canonical in _COWORKING_OPERATORS:
        if needle in lowered:
            return canonical

    return None


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def _ascii_fold(text: str) -> str:
    return (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode("ascii")
    )


def clean_text(value: object) -> str | None:
    """Trim/collapse whitespace; convert empty-ish values to None."""
    if value is None:
        return None

    s = _WS.sub(" ", str(value)).strip()

    return (
        None
        if s.lower() in {"", "nan", "none", "null", "n/a", "na", "-"}
        else s
    )


def normalize_company_name(name: str) -> str:
    """
    Lowercase, ASCII-fold, strip punctuation and legal/generic suffixes.

    Examples:
        ABC Technologies Pvt Ltd -> abc
        ABC Tech                 -> abc
        3M India Ltd             -> 3m

    Falls back to the un-suffix-stripped form if stripping would leave
    nothing useful.
    """
    folded = _ascii_fold(name).lower().replace("&", " and ")

    stripped = _SUFFIX_RE.sub(" ", folded)

    cleaned = _WS.sub(
        " ",
        _NON_ALNUM.sub(" ", stripped),
    ).strip()

    if cleaned:
        return cleaned

    return _WS.sub(
        " ",
        _NON_ALNUM.sub(" ", folded),
    ).strip()


# ---------------------------------------------------------------------------
# URL / domain normalization
# ---------------------------------------------------------------------------

def normalize_url(value: object) -> str | None:
    """
    Return a canonical https URL, or None if unusable.

    Rejects non-http(s) schemes.
    """
    s = clean_text(value)

    if not s:
        return None

    if "://" not in s:
        s = "https://" + s

    try:
        parsed = urlparse(s)
    except ValueError:
        return None

    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or "." not in parsed.hostname
    ):
        return None

    host = parsed.hostname.lower()
    path = parsed.path.rstrip("/")

    return f"https://{host}{path}" if path else f"https://{host}"


def extract_domain(url_or_domain: object) -> str | None:
    """Host without www. Does not compute registrable domain."""
    url = normalize_url(url_or_domain)

    if not url:
        return None

    host = urlparse(url).hostname or ""

    return host[4:] if host.startswith("www.") else host or None


def registrable_hint(domain: str | None) -> str | None:
    """
    Best-effort registrable domain used ONLY as a blocking key,
    never as proof of identity.
    """
    if not domain:
        return None

    parts = domain.split(".")

    if (
        len(parts) >= 3
        and ".".join(parts[-2:]) in _TWO_PART_SUFFIXES
    ):
        return ".".join(parts[-3:])

    return ".".join(parts[-2:]) if len(parts) >= 2 else domain


# ---------------------------------------------------------------------------
# LinkedIn
# ---------------------------------------------------------------------------

def normalize_linkedin_url(value: object) -> str | None:
    """
    Canonical LinkedIn company URL or None.
    """
    url = normalize_url(value)

    if not url:
        return None

    parsed = urlparse(url)
    host = (parsed.hostname or "").removeprefix("www.")

    if host != "linkedin.com":
        return None

    match = re.match(
        r"^/company/([A-Za-z0-9\-_%.]+)",
        parsed.path,
    )

    if not match:
        return None

    slug = match.group(1).lower()

    return f"https://www.linkedin.com/company/{slug}"


# ---------------------------------------------------------------------------
# Phone
# ---------------------------------------------------------------------------

def normalize_phone(
    value: object,
    default_region: str = "IN",
) -> str | None:
    """
    E.164 string if the number parses AND is valid.
    Otherwise None.
    """
    s = clean_text(value)

    if not s:
        return None

    try:
        num = phonenumbers.parse(
            s,
            default_region,
        )
    except phonenumbers.NumberParseException:
        return None

    if not phonenumbers.is_valid_number(num):
        return None

    return phonenumbers.format_number(
        num,
        phonenumbers.PhoneNumberFormat.E164,
    )


# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------

def normalize_email(value: object) -> str | None:
    s = clean_text(value)

    if not s:
        return None

    s = s.lower()

    return (
        s
        if re.fullmatch(
            r"[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}",
            s,
        )
        else None
    )


# ---------------------------------------------------------------------------
# City
# ---------------------------------------------------------------------------

def normalize_city(value: object) -> str | None:
    s = clean_text(value)

    if not s:
        return None

    key = _WS.sub(" ", s.lower())

    return _CITY_ALIASES.get(
        key,
        s.title(),
    )


# ---------------------------------------------------------------------------
# Numeric parsers
# ---------------------------------------------------------------------------

def parse_int(value: object) -> int | None:
    """
    Parse values such as:
        1,200
        1200.0
        ~50

    Returns None when not clearly numeric.
    """
    s = clean_text(value)

    if s is None:
        return None

    s = s.replace(",", "").lstrip("~")

    try:
        f = float(s)
    except ValueError:
        return None

    return (
        int(f)
        if f >= 0 and f == int(f)
        else None
    )


def parse_float(value: object) -> float | None:
    s = clean_text(value)

    if s is None:
        return None

    try:
        return float(s)
    except ValueError:
        return None