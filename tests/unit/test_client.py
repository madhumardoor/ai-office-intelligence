import pytest

from app.llm.cache import LRUCache
from app.llm.client import LLMClient
from app.llm.mock import MockLLM
from app.llm.pricing import PriceTable
from app.llm.types import (
    LLMOutputError,
    LLMPermanentError,
    LLMUnavailableError,
    Tier,
)
from app.llm.usage import UsageRecorder
from app.schemas.llm_outputs import RouteDecision


async def _nosleep(_: float) -> None:
    return None


def _client(mock: MockLLM, **kwargs) -> LLMClient:  # type: ignore[no-untyped-def]
    return LLMClient(
        mock,
        sleep=_nosleep,
        **kwargs,
    )


async def test_tier_selects_model() -> None:
    mock = MockLLM()
    client = _client(mock)

    await client.generate_text(
        purpose="test",
        tier=Tier.FAST,
        system="s",
        user="u",
    )

    await client.generate_text(
        purpose="test",
        tier=Tier.STRONG,
        system="s",
        user="u2",
    )

    assert [call["tier"] for call in mock.calls] == [
        Tier.FAST,
        Tier.STRONG,
    ]

    assert client.model_for(Tier.FAST) == "mock-fast"
    assert client.model_for(Tier.STRONG) == "mock-strong"


async def test_deterministic_calls_are_cached() -> None:
    mock = MockLLM()

    client = _client(
        mock,
        cache=LRUCache(10),
    )

    first = await client.generate_text(
        purpose="test",
        tier=Tier.FAST,
        system="s",
        user="u",
    )

    second = await client.generate_text(
        purpose="test",
        tier=Tier.FAST,
        system="s",
        user="u",
    )

    assert first == second
    assert len(mock.calls) == 1


async def test_nonzero_temperature_is_never_cached() -> None:
    mock = MockLLM()

    client = _client(
        mock,
        cache=LRUCache(10),
    )

    for _ in range(2):
        await client.generate_text(
            purpose="test",
            tier=Tier.FAST,
            system="s",
            user="u",
            temperature=0.7,
        )

    assert len(mock.calls) == 2


async def test_different_tier_or_prompt_does_not_share_cache() -> None:
    mock = MockLLM()

    client = _client(
        mock,
        cache=LRUCache(10),
    )

    await client.generate_text(
        purpose="test",
        tier=Tier.FAST,
        system="s",
        user="u",
    )

    await client.generate_text(
        purpose="test",
        tier=Tier.STRONG,
        system="s",
        user="u",
    )

    await client.generate_text(
        purpose="test",
        tier=Tier.FAST,
        system="s",
        user="other",
    )

    assert len(mock.calls) == 3


async def test_transient_failures_retried_transparently() -> None:
    mock = MockLLM(fail_times=2)

    result = await _client(
        mock,
        max_retries=3,
    ).generate_text(
        purpose="test",
        tier=Tier.FAST,
        system="s",
        user="u",
    )

    assert result == "mock answer"
    assert len(mock.calls) == 3


async def test_permanent_error_propagates_immediately() -> None:
    mock = MockLLM(
        fail_times=1,
        exc_factory=lambda: LLMPermanentError("401"),
    )

    with pytest.raises(LLMPermanentError):
        await _client(
            mock,
            max_retries=3,
        ).generate_text(
            purpose="test",
            tier=Tier.FAST,
            system="s",
            user="u",
        )

    assert len(mock.calls) == 1


async def test_outage_raises_unavailable_never_fabricates_text() -> None:
    mock = MockLLM(fail_times=99)

    with pytest.raises(LLMUnavailableError):
        await _client(
            mock,
            max_retries=1,
        ).generate_text(
            purpose="test",
            tier=Tier.FAST,
            system="s",
            user="u",
        )


async def test_structured_returns_validated_model() -> None:
    mock = MockLLM(
        structured_fn=lambda system, user, schema: {
            "query_type": "postgis",
            "tools_required": [
                "postgis_search",
                "postgis_search",
            ],
        }
    )

    result = await _client(mock).generate_structured(
        purpose="route",
        tier=Tier.FAST,
        system="s",
        user="u",
        schema=RouteDecision,
    )

    assert isinstance(result, RouteDecision)

    assert result.tools_required == [
        "postgis_search",
    ]


async def test_structured_repair_attempt_then_success() -> None:
    calls = {"count": 0}

    def structured_fn(
        system,
        user,
        schema,
    ):  # type: ignore[no-untyped-def]
        calls["count"] += 1

        if calls["count"] == 1:
            raise LLMOutputError("bad json")

        assert "could not be parsed" in user

        return {
            "query_type": "vector",
        }

    mock = MockLLM(
        structured_fn=structured_fn,
    )

    result = await _client(mock).generate_structured(
        purpose="route",
        tier=Tier.FAST,
        system="s",
        user="u",
        schema=RouteDecision,
    )

    assert result.query_type == "vector"
    assert calls["count"] == 2


async def test_structured_gives_up_after_one_repair() -> None:
    def structured_fn(
        system,
        user,
        schema,
    ):  # type: ignore[no-untyped-def]
        raise LLMOutputError("still bad")

    mock = MockLLM(
        structured_fn=structured_fn,
    )

    with pytest.raises(LLMOutputError):
        await _client(mock).generate_structured(
            purpose="route",
            tier=Tier.FAST,
            system="s",
            user="u",
            schema=RouteDecision,
        )

    assert len(mock.calls) == 2


async def test_invalid_structured_data_is_rejected() -> None:
    mock = MockLLM(
        structured_fn=lambda system, user, schema: {
            "query_type": "drop_all_tables",
        }
    )

    with pytest.raises(Exception):
        await _client(mock).generate_structured(
            purpose="route",
            tier=Tier.FAST,
            system="s",
            user="u",
            schema=RouteDecision,
        )


async def test_usage_recorder_tracks_cost_and_skips_cache_hits() -> None:
    prices = PriceTable.from_json(
        '{"mock-fast": {"input_per_million": 1000000, '
        '"output_per_million": 1000000}}'
    )

    recorder = UsageRecorder(
        None,
        enabled=False,
    )

    client = _client(
        MockLLM(),
        prices=prices,
        recorder=recorder,
        cache=LRUCache(5),
    )

    await client.generate_text(
        purpose="test",
        tier=Tier.FAST,
        system="s",
        user="u",
    )

    cost_after_first = (
        recorder.session_total_cost_usd
    )

    await client.generate_text(
        purpose="test",
        tier=Tier.FAST,
        system="s",
        user="u",
    )

    assert cost_after_first > 0
    assert (
        recorder.session_total_cost_usd
        == cost_after_first
    )
    assert recorder.session_calls == 2