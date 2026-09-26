import asyncio

import pytest

from app.llm.resilience import (
    CircuitBreaker,
    call_with_resilience,
)
from app.llm.types import (
    LLMPermanentError,
    LLMTransientError,
    LLMUnavailableError,
)


async def _nosleep(_: float) -> None:
    return None


async def test_retries_transient_then_succeeds() -> None:
    attempts = {"n": 0}

    async def fn() -> str:
        attempts["n"] += 1

        if attempts["n"] < 3:
            raise LLMTransientError("429")

        return "ok"

    result = await call_with_resilience(
        fn,
        breaker=CircuitBreaker(),
        timeout_s=5,
        max_retries=3,
        sleep=_nosleep,
    )

    assert result == "ok"
    assert attempts["n"] == 3


async def test_permanent_error_not_retried() -> None:
    attempts = {"n": 0}

    async def fn() -> str:
        attempts["n"] += 1
        raise LLMPermanentError("401")

    with pytest.raises(LLMPermanentError):
        await call_with_resilience(
            fn,
            breaker=CircuitBreaker(),
            timeout_s=5,
            max_retries=3,
            sleep=_nosleep,
        )

    assert attempts["n"] == 1


async def test_exhausted_retries_raise_unavailable() -> None:
    async def fn() -> str:
        raise LLMTransientError("503")

    with pytest.raises(LLMUnavailableError):
        await call_with_resilience(
            fn,
            breaker=CircuitBreaker(),
            timeout_s=5,
            max_retries=2,
            sleep=_nosleep,
        )


async def test_timeout_counts_as_transient() -> None:
    async def fn() -> str:
        await asyncio.sleep(1)
        return "late"

    with pytest.raises(LLMUnavailableError):
        await call_with_resilience(
            fn,
            breaker=CircuitBreaker(),
            timeout_s=0.01,
            max_retries=1,
            sleep=_nosleep,
        )


async def test_breaker_opens_then_fails_fast_then_half_opens() -> None:
    now = {"t": 0.0}

    breaker = CircuitBreaker(
        failure_threshold=2,
        reset_seconds=30,
        clock=lambda: now["t"],
    )

    calls = {"n": 0}

    async def bad() -> str:
        calls["n"] += 1
        raise LLMTransientError("down")

    for _ in range(2):
        with pytest.raises(LLMUnavailableError):
            await call_with_resilience(
                bad,
                breaker=breaker,
                timeout_s=5,
                max_retries=0,
                sleep=_nosleep,
            )

    before = calls["n"]

    with pytest.raises(
        LLMUnavailableError,
        match="circuit",
    ):
        await call_with_resilience(
            bad,
            breaker=breaker,
            timeout_s=5,
            max_retries=0,
            sleep=_nosleep,
        )

    assert calls["n"] == before

    now["t"] = 31.0

    async def good() -> str:
        return "back"

    assert await call_with_resilience(
        good,
        breaker=breaker,
        timeout_s=5,
        max_retries=0,
        sleep=_nosleep,
    ) == "back"

    assert not breaker.is_open