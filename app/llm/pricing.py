"""Cost estimation from a user-supplied price table.

Never invents prices.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Price:
    input_per_million: float
    output_per_million: float


class PriceTable:
    """Maps model IDs to user-configured input/output prices."""

    def __init__(self, prices: dict[str, Price] | None = None) -> None:
        self._prices = prices or {}

    @classmethod
    def from_json(cls, raw: str) -> "PriceTable":
        """Create a price table from JSON.

        Invalid or empty configuration disables cost estimation
        rather than breaking the application.
        """
        if not raw.strip():
            return cls()

        try:
            data = json.loads(raw)

            prices = {
                model: Price(
                    input_per_million=float(value["input_per_million"]),
                    output_per_million=float(value["output_per_million"]),
                )
                for model, value in data.items()
            }

            return cls(prices)

        except (ValueError, KeyError, TypeError, AttributeError, json.JSONDecodeError) as exc:
            logger.error(
                "invalid LLM_PRICING_JSON; cost estimation disabled",
                extra={"error": type(exc).__name__},
            )
            return cls()

    def estimate(
        self,
        model: str,
        input_tokens: int | None,
        output_tokens: int | None,
    ) -> float | None:
        """Return estimated USD cost, or None when unavailable."""
        price = self._prices.get(model)

        if price is None:
            return None

        if input_tokens is None or output_tokens is None:
            return None

        return round(
            (
                input_tokens * price.input_per_million
                + output_tokens * price.output_per_million
            )
            / 1_000_000,
            6,
        )