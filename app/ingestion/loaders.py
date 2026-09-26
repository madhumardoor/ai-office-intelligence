"""Load CSV/Excel into canonical-keyed dict rows. Structural problems raise; row problems are collected."""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from app.ingestion.schemas import COLUMN_ALIASES, REQUIRED_ANY_OF

SUPPORTED_SUFFIXES = {".csv", ".xlsx", ".xls"}
MAX_ROWS = 200_000  # guardrail against accidental huge uploads


class IngestionFileError(Exception):
    """The file itself is unusable (wrong type, no required columns, unreadable)."""


@dataclass(slots=True)
class LoadedFile:
    path: Path
    rows: list[tuple[int, dict[str, object]]]  # (1-based data row number, canonical dict)
    unknown_columns: list[str] = field(default_factory=list)


def _canon_header(h: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(h).strip().lower()).strip("_")


def _map_columns(columns: list[object]) -> tuple[dict[str, str], list[str]]:
    """Return ({original_header: canonical_name}, [unrecognised headers])."""
    lookup = {alias: canon for canon, aliases in COLUMN_ALIASES.items() for alias in aliases}
    mapping: dict[str, str] = {}
    unknown: list[str] = []
    seen: set[str] = set()
    for col in columns:
        canon = lookup.get(_canon_header(col))
        if canon is None:
            unknown.append(str(col))
        elif canon in seen:
            raise IngestionFileError(f"duplicate column for '{canon}': {col!r}")
        else:
            mapping[str(col)] = canon
            seen.add(canon)
    return mapping, unknown


def load_file(path: str | Path) -> LoadedFile:
    p = Path(path)
    if not p.is_file():
        raise IngestionFileError(f"file not found: {p}")
    if p.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise IngestionFileError(f"unsupported file type {p.suffix!r}; use .csv or .xlsx")
    try:
        df = (
            pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8-sig")
            if p.suffix.lower() == ".csv"
            else pd.read_excel(p, dtype=str, keep_default_na=False)
        )
    except Exception as exc:  # noqa: BLE001 - surface any parser failure as a file error
        raise IngestionFileError(f"could not parse {p.name}: {exc}") from exc

    if len(df) > MAX_ROWS:
        raise IngestionFileError(f"file has {len(df)} rows; limit is {MAX_ROWS}")

    mapping, unknown = _map_columns(list(df.columns))
    canon_cols = set(mapping.values())
    if not canon_cols & set(REQUIRED_ANY_OF):
        raise IngestionFileError(f"file must contain one of these columns: {', '.join(REQUIRED_ANY_OF)}")

    df = df.rename(columns=mapping)[list(mapping.values())]
    rows = [(i + 1, rec) for i, rec in enumerate(df.to_dict(orient="records"))]
    return LoadedFile(path=p, rows=rows, unknown_columns=unknown)


def iter_from(loaded: LoadedFile, start_after_row: int = 0) -> Iterator[tuple[int, dict[str, object]]]:
    """Yield rows after a checkpoint (resume support)."""
    for num, rec in loaded.rows:
        if num > start_after_row:
            yield num, rec