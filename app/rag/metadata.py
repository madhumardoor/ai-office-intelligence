"""Best-effort metadata extraction (dates, location). Never invents values: returns None when unsure."""

from __future__ import annotations

import re
from datetime import date

_MONTHS = {
    m: i
    for i, m in enumerate(
        ["january", "february", "march", "april", "may", "june", "july", "august",
         "september", "october", "november", "december"],
        start=1,
    )
}
_MONTH_ALT = "|".join(_MONTHS)

_ISO = re.compile(r"(?<!\d)(20\d{2})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])(?!\d)")
_LONG = re.compile(rf"\b({_MONTH_ALT})\s+(\d{{1,2}}),?\s+(20\d{{2}})\b", re.IGNORECASE)
_LONG_DMY = re.compile(rf"\b(\d{{1,2}})\s+({_MONTH_ALT}),?\s+(20\d{{2}})\b", re.IGNORECASE)

KNOWN_LOCATIONS = (
    "Bengaluru", "Bangalore", "Whitefield", "Koramangala", "Indiranagar", "HSR Layout",
    "Electronic City", "Bellandur", "Marathahalli", "Hebbal", "Manyata",
)


def _safe_date(y: int, m: int, d: int) -> date | None:
    try:
        return date(y, m, d)
    except ValueError:
        return None


def extract_published_date(raw_hint: str | None, text: str) -> date | None:
    """Prefer an explicit metadata hint; else the first plausible date in the first 600 characters
    (publication dates appear near the top; later dates are usually events mentioned in the body)."""
    for source in (raw_hint or "", text[:600]):
        if m := _ISO.search(source):
            if d := _safe_date(int(m[1]), int(m[2]), int(m[3])):
                return d
        if m := _LONG.search(source):
            if d := _safe_date(int(m[3]), _MONTHS[m[1].lower()], int(m[2])):
                return d
        if m := _LONG_DMY.search(source):
            if d := _safe_date(int(m[3]), _MONTHS[m[2].lower()], int(m[1])):
                return d
    return None


def extract_location(text: str) -> str | None:
    """Most frequently mentioned known location; ties resolved by first appearance."""
    lower = text.lower()
    counts = [
        (lower.count(loc.lower()), -lower.find(loc.lower()), loc)
        for loc in KNOWN_LOCATIONS
        if loc.lower() in lower
    ]
    if not counts:
        return None
    counts.sort(reverse=True)
    return counts[0][2]