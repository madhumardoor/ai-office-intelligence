"""Small in-process LRU cache for deterministic LLM calls."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Any


def make_key(*parts: object) -> str:
    """Create a collision-resistant cache key."""
    h = hashlib.sha256()

    for part in parts:
        h.update(repr(part).encode("utf-8"))
        h.update(b"\x1f")

    return h.hexdigest()


class LRUCache:
    """Simple bounded in-process LRU cache."""

    def __init__(self, max_entries: int = 500) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")

        self._max = max_entries
        self._data: OrderedDict[str, Any] = OrderedDict()

    def get(self, key: str) -> Any | None:
        """Return cached value and mark it as recently used."""
        if key not in self._data:
            return None

        self._data.move_to_end(key)
        return self._data[key]

    def put(self, key: str, value: Any) -> None:
        """Insert/update a cached value."""
        self._data[key] = value
        self._data.move_to_end(key)

        while len(self._data) > self._max:
            self._data.popitem(last=False)

    def __len__(self) -> int:
        return len(self._data)

    def clear(self) -> None:
        """Clear all cached values."""
        self._data.clear()