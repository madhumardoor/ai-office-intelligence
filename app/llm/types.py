"""Provider-independent LLM contracts.

Nothing in this file imports LangChain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, TypeVar

from pydantic import BaseModel


T = TypeVar("T", bound=BaseModel)


class Tier(StrEnum):
    """FAST is for routing/extraction; STRONG is for final reasoning."""

    FAST = "fast"
    STRONG = "strong"


class LLMError(Exception):
    """Base class for all LLM-layer errors."""


class LLMTransientError(LLMError):
    """Timeouts, rate limits, 5xx, and temporary network failures."""


class LLMPermanentError(LLMError):
    """Authentication, invalid request, unsupported model, or safety failures."""


class LLMOutputError(LLMError):
    """The model answered but structured output could not be validated."""


class LLMUnavailableError(LLMError):
    """The LLM is unavailable after bounded retries or circuit breaking."""


@dataclass(frozen=True, slots=True)
class Usage:
    """Token usage reported by a provider."""

    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass(slots=True)
class LLMResult:
    """Raw text-generation result."""

    text: str
    usage: Usage = field(default_factory=Usage)


@dataclass(slots=True)
class StructuredResult[T: BaseModel]:
    """Validated structured-generation result."""

    value: T
    usage: Usage = field(default_factory=Usage)


@dataclass(slots=True)
class CallMeta:
    """Observability metadata for one logical LLM call."""

    purpose: str
    tier: Tier
    provider: str
    model: str
    latency_ms: float
    input_tokens: int | None
    output_tokens: int | None
    estimated_cost_usd: float | None
    cache_hit: bool
    prompt_hash: str
    error: str | None = None


class ProviderAdapter(Protocol):
    """Provider-independent interface implemented by Gemini/OpenAI/mock."""

    provider: str

    def model_for(self, tier: Tier) -> str:
        ...

    async def text(
        self,
        tier: Tier,
        system: str,
        user: str,
        *,
        max_tokens: int,
        temperature: float,
    ) -> LLMResult:
        ...

    async def structured(
        self,
        tier: Tier,
        system: str,
        user: str,
        schema: type[T],
        *,
        max_tokens: int,
        temperature: float,
    ) -> StructuredResult[T]:
        ...