"""Provider-agnostic LLM reliability.

Transient failures are retried with bounded exponential backoff.
Permanent failures are not retried.
A circuit breaker prevents repeated calls during provider outages.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

from app.llm.types import (
    LLMPermanentError,
    LLMTransientError,
    LLMUnavailableError,
)


logger = logging.getLogger(__name__)

R = TypeVar("R")


class CircuitBreaker:
    """Small in-process circuit breaker."""

    def __init__(
        self,
        failure_threshold: int = 5,
        reset_seconds: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")

        if reset_seconds <= 0:
            raise ValueError("reset_seconds must be > 0")

        self._threshold = failure_threshold
        self._reset = reset_seconds
        self._clock = clock
        self._failures = 0
        self._opened_at: float | None = None

    @property
    def is_open(self) -> bool:
        """Return whether calls should currently fail fast."""
        if self._opened_at is None:
            return False

        if self._clock() - self._opened_at >= self._reset:
            # Half-open state: allow one trial call.
            return False

        return True

    def record_success(self) -> None:
        """Reset breaker after a successful call."""
        self._failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        """Record a failed logical call."""
        self._failures += 1

        if self._failures >= self._threshold:
            self._opened_at = self._clock()


async def call_with_resilience(
    fn: Callable[[], Awaitable[R]],
    *,
    breaker: CircuitBreaker,
    timeout_s: float,
    max_retries: int,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    base_delay: float = 1.0,
    max_delay: float = 20.0,
) -> R:
    """Execute an async provider call with timeout/retry/circuit breaking."""

    if timeout_s <= 0:
        raise ValueError("timeout_s must be > 0")

    if max_retries < 0:
        raise ValueError("max_retries must be >= 0")

    if breaker.is_open:
        raise LLMUnavailableError(
            "LLM circuit breaker is open; failing fast"
        )

    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            result = await asyncio.wait_for(fn(), timeout=timeout_s)
            breaker.record_success()
            return result

        except LLMPermanentError:
            breaker.record_failure()
            raise

        except (LLMTransientError, TimeoutError) as exc:
            last_error = exc

            if attempt == max_retries:
                break

            delay = min(
                max_delay,
                base_delay * (2**attempt),
            ) * random.uniform(0.5, 1.0)

            logger.warning(
                "LLM transient failure; retrying",
                extra={
                    "attempt": attempt + 1,
                    "delay_s": round(delay, 2),
                    "error": type(exc).__name__,
                },
            )

            await sleep(delay)

    breaker.record_failure()

    raise LLMUnavailableError(
        f"LLM unavailable after {max_retries + 1} attempts: "
        f"{type(last_error).__name__}"
    ) from last_error