"""Concrete LLM provider adapters.

THIS IS THE ONLY MODULE THAT IMPORTS LANGCHAIN.

Everything else depends on the provider-independent ProviderAdapter
protocol from app.llm.types.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, ValidationError

from app.config import Settings
from app.llm.types import (
    LLMOutputError,
    LLMPermanentError,
    LLMResult,
    LLMTransientError,
    StructuredResult,
    Tier,
    Usage,
)


logger = logging.getLogger(__name__)


_PERMANENT_HINTS = (
    "401",
    "403",
    "400",
    "404",
    "invalid api key",
    "api key not valid",
    "authentication",
    "permission denied",
    "permission",
    "unauthenticated",
    "not found",
    "invalid argument",
    "unsupported",
    "safety",
    "blocked",
    "model not found",
)

_TRANSIENT_HINTS = (
    "408",
    "429",
    "500",
    "502",
    "503",
    "504",
    "rate limit",
    "resource exhausted",
    "quota",
    "timeout",
    "timed out",
    "connection",
    "unavailable",
    "overloaded",
    "reset by peer",
)


def classify_exception(exc: Exception) -> Exception:
    """Map provider exceptions into our own error types."""
    status = (
        getattr(exc, "status_code", None)
        or getattr(exc, "status", None)
        or getattr(
            getattr(exc, "response", None),
            "status_code",
            None,
        )
    )

    text = f"{type(exc).__name__} {exc}".lower()

    if isinstance(status, int):
        if status == 408 or status == 429 or status >= 500:
            return LLMTransientError(
                f"HTTP {status}"
            )

        if 400 <= status < 500:
            return LLMPermanentError(
                f"HTTP {status}"
            )

    if any(hint in text for hint in _TRANSIENT_HINTS):
        return LLMTransientError(
            type(exc).__name__
        )

    if any(hint in text for hint in _PERMANENT_HINTS):
        return LLMPermanentError(
            type(exc).__name__
        )

    # Unknown provider errors are treated as transient so the
    # outer resilience layer gets a bounded retry opportunity.
    return LLMTransientError(
        f"unclassified provider error: {type(exc).__name__}"
    )


def _extract_text(content: Any) -> str:
    """Normalize provider-specific AIMessage.content shapes."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        pieces: list[str] = []

        for part in content:
            if isinstance(part, str):
                pieces.append(part)
            elif isinstance(part, dict):
                text = part.get("text")
                if text:
                    pieces.append(str(text))

        return "".join(pieces)

    return str(content)


def _usage_from(message: Any) -> Usage:
    """Extract token counts from LangChain usage_metadata."""
    metadata = getattr(message, "usage_metadata", None) or {}

    return Usage(
        input_tokens=metadata.get("input_tokens"),
        output_tokens=metadata.get("output_tokens"),
    )


class LangChainAdapter:
    """Wrap Gemini/OpenAI behind the ProviderAdapter contract."""

    def __init__(self, settings: Settings) -> None:
        if not settings.llm_model_fast:
            raise LLMPermanentError(
                "LLM_MODEL_FAST must be set"
            )

        if not settings.llm_model_strong:
            raise LLMPermanentError(
                "LLM_MODEL_STRONG must be set"
            )

        if not settings.llm_api_key.get_secret_value():
            raise LLMPermanentError(
                "LLM_API_KEY is not set"
            )

        self.provider = settings.llm_provider
        self._settings = settings

        self._models: dict[
            tuple[Tier, float, int],
            Any,
        ] = {}

    def model_for(self, tier: Tier) -> str:
        if tier is Tier.FAST:
            return self._settings.llm_model_fast

        return self._settings.llm_model_strong

    def _chat(
        self,
        tier: Tier,
        temperature: float,
        max_tokens: int,
    ) -> Any:
        """Create/cache one configured LangChain model."""

        cache_key = (
            tier,
            temperature,
            max_tokens,
        )

        existing = self._models.get(cache_key)

        if existing is not None:
            return existing

        api_key = (
            self._settings.llm_api_key
            .get_secret_value()
        )

        model_name = self.model_for(tier)

        if self.provider == "gemini":
            from langchain_google_genai import (
                ChatGoogleGenerativeAI,
            )

            model = ChatGoogleGenerativeAI(
                model=model_name,
                api_key=api_key,
                temperature=temperature,
                max_output_tokens=max_tokens,
                max_retries=0,
            )

        elif self.provider == "openai":
            from langchain_openai import ChatOpenAI

            model = ChatOpenAI(
                model=model_name,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                max_retries=0,
            )
            
        elif self.provider == "groq":
            from langchain_groq import ChatGroq

            model = ChatGroq(
            model=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=0,
           )
        else:
            raise LLMPermanentError(
                f"unsupported LLM provider: {self.provider!r}"
            )

        self._models[cache_key] = model
        return model

    @staticmethod
    def _messages(
        system: str,
        user: str,
    ) -> list[Any]:
        from langchain_core.messages import (
            HumanMessage,
            SystemMessage,
        )

        return [
            SystemMessage(content=system),
            HumanMessage(content=user),
        ]

    async def text(
        self,
        tier: Tier,
        system: str,
        user: str,
        *,
        max_tokens: int,
        temperature: float,
    ) -> LLMResult:
        try:
            message = await self._chat(
                tier,
                temperature,
                max_tokens,
            ).ainvoke(
                self._messages(system, user)
            )

        except Exception as exc:
            raise classify_exception(exc) from exc

        return LLMResult(
            text=_extract_text(message.content),
            usage=_usage_from(message),
        )

    async def structured(
        self,
        tier: Tier,
        system: str,
        user: str,
        schema: type[BaseModel],
        *,
        max_tokens: int,
        temperature: float,
    ) -> StructuredResult[Any]:

        chat = self._chat(
            tier,
            temperature,
            max_tokens,
        )

        try:
            runnable = chat.with_structured_output(
                schema,
                include_raw=True,
            )

            output = await runnable.ainvoke(
                self._messages(system, user)
            )

        except Exception as exc:
            raise classify_exception(exc) from exc

        if not isinstance(output, dict):
            raise LLMOutputError(
                "structured provider returned an unexpected "
                f"type: {type(output).__name__}"
            )

        raw = output.get("raw")
        parsed = output.get("parsed")
        parsing_error = output.get("parsing_error")

        usage = (
            _usage_from(raw)
            if raw is not None
            else Usage()
        )

        if isinstance(parsed, BaseModel):
            return StructuredResult(
                value=parsed,
                usage=usage,
            )

        if isinstance(parsed, dict):
            try:
                value = schema.model_validate(parsed)
            except ValidationError as exc:
                raise LLMOutputError(
                    "structured output failed validation: "
                    f"{exc.error_count()} error(s)"
                ) from exc

            return StructuredResult(
                value=value,
                usage=usage,
            )

        raise LLMOutputError(
            "model returned no parseable structured output "
            f"({type(parsing_error).__name__ if parsing_error else 'empty'})"
        )