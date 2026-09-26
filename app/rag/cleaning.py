"""Deterministic text cleaning. Conservative: removes noise, never rewrites meaning."""

from __future__ import annotations

import hashlib
import re
import unicodedata

_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MULTI_BLANK = re.compile(r"\n{3,}")
_SPACES = re.compile(r"[ \t\u00a0]+")
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")  # PDF line-break hyphenation: "expan-\nsion"
_PAGE_NUM = re.compile(r"^\s*(?:page\s+)?\d{1,4}(?:\s*(?:of|/)\s*\d{1,4})?\s*$", re.IGNORECASE)


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _CTRL.sub("", text)
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    lines = [
        ln for ln in (_SPACES.sub(" ", ln).strip() for ln in text.splitlines()) if not _PAGE_NUM.match(ln)
    ]
    text = "\n".join(lines)
    text = _MULTI_BLANK.sub("\n\n", text)
    return text.strip()


def content_hash(text: str) -> str:
    """Stable SHA-256 over whitespace-normalised, case-preserved text (case can carry meaning)."""
    normalised = re.sub(r"\s+", " ", text).strip()
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()