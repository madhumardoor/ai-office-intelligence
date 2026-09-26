from app.llm.cache import LRUCache, make_key


def test_lru_eviction_and_recency() -> None:
    cache = LRUCache(2)

    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a")
    cache.put("c", 3)

    assert cache.get("b") is None
    assert cache.get("a") == 1
    assert cache.get("c") == 3
    assert len(cache) == 2


def test_key_separator_prevents_collisions() -> None:
    assert make_key("ab", "c") != make_key("a", "bc")


def test_miss_returns_none() -> None:
    assert LRUCache().get("x") is None