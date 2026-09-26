"""Structure-aware chunking.

Strategy: split into paragraphs; oversized paragraphs are split on sentences; pack units into chunks up
to a token target; carry a tail of the previous chunk as overlap so facts spanning a boundary survive.
Token counts are APPROXIMATE (~4 chars/token); chunk sizing doesn't need to be model-exact.
"""

from __future__ import annotations

import re

from app.rag.cleaning import content_hash
from app.rag.types import Chunk

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
CHARS_PER_TOKEN = 4


def approx_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def _units(text: str, target: int) -> list[str]:
    """Break text into atomic units no larger than `target` tokens."""
    units: list[str] = []
    for para in (p.strip() for p in re.split(r"\n\s*\n", text)):
        if not para:
            continue
        if approx_tokens(para) <= target:
            units.append(para)
            continue
        for sent in _SENT_SPLIT.split(para):
            sent = sent.strip()
            if not sent:
                continue
            if approx_tokens(sent) <= target:
                units.append(sent)
            else:  # pathological: giant unbroken string. Hard-split by characters.
                step = target * CHARS_PER_TOKEN
                units.extend(sent[i : i + step] for i in range(0, len(sent), step))
    return units


def _overlap_tail(text: str, overlap_tokens: int) -> str:
    """Last ~overlap_tokens of text, snapped forward to a word boundary."""
    if overlap_tokens <= 0:
        return ""
    tail = text[-overlap_tokens * CHARS_PER_TOKEN :]
    space = tail.find(" ")
    return tail[space + 1 :] if 0 <= space < len(tail) - 1 else tail


def chunk_text(text: str, target_tokens: int = 450, overlap_tokens: int = 60) -> list[Chunk]:
    if overlap_tokens >= target_tokens:
        raise ValueError("overlap_tokens must be smaller than target_tokens")
    units = _units(text, target_tokens)
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    def flush() -> None:
        nonlocal current, current_tokens
        if current:
            chunks.append("\n\n".join(current))
        current, current_tokens = [], 0

    for unit in units:
        ut = approx_tokens(unit)
        if current and current_tokens + ut > target_tokens:
            prev = "\n\n".join(current)
            flush()
            tail = _overlap_tail(prev, overlap_tokens)
            if tail and approx_tokens(tail) + ut <= target_tokens:
                current, current_tokens = [tail], approx_tokens(tail)
        current.append(unit)
        current_tokens += ut
    flush()

    return [
        Chunk(index=i, content=c, content_hash=content_hash(c), token_count=approx_tokens(c))
        for i, c in enumerate(chunks)
    ]