from __future__ import annotations

import time
from collections import OrderedDict
from copy import deepcopy
from typing import Any


class SearchCache:
    """
    Small bounded TTL cache for web/news search results.
    """

    def __init__(
        self,
        max_entries: int = 100,
        ttl_seconds: float = 300.0,
    ) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")

        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be > 0")

        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._store: OrderedDict[
            tuple[Any, ...],
            tuple[float, Any],
        ] = OrderedDict()

    def get(self, key: tuple[Any, ...]) -> Any | None:
        entry = self._store.get(key)

        if entry is None:
            return None

        created_at, value = entry

        if time.monotonic() - created_at >= self.ttl_seconds:
            self._store.pop(key, None)
            return None

        self._store.move_to_end(key)

        return deepcopy(value)

    def set(
        self,
        key: tuple[Any, ...],
        value: Any,
    ) -> None:
        self._store[key] = (time.monotonic(), deepcopy(value))
        self._store.move_to_end(key)

        while len(self._store) > self.max_entries:
            self._store.popitem(last=False)

    def clear(self) -> None:
        self._store.clear()

    @property
    def size(self) -> int:
        return len(self._store)