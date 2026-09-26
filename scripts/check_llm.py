"""Live smoke test for the configured LLM provider.

Run:
    uv run python -m scripts.check_llm

This makes:
    1. one FAST text call
    2. one STRONG text call
    3. one FAST structured-output call
"""

from __future__ import annotations

import asyncio
import sys
import traceback

from pydantic import BaseModel

from app.config import get_settings
from app.llm.factory import create_llm_client
from app.llm.types import Tier


class Ping(BaseModel):
    """Minimal structured-output smoke-test schema."""

    ok: bool
    word: str


async def main() -> int:
    settings = get_settings()

    print(
        f"provider={settings.llm_provider} "
        f"fast={settings.llm_model_fast!r} "
        f"strong={settings.llm_model_strong!r}"
    )

    try:
        client = create_llm_client(settings)

    except Exception as exc:
        print(
            "FAIL config/adapter construction: "
            f"{type(exc).__name__}: {exc}"
        )
        return 1

    status = 0

    for tier in (
        Tier.FAST,
        Tier.STRONG,
    ):
        try:
            output = await client.generate_text(
                purpose="smoke",
                tier=tier,
                system="Reply with exactly one word.",
                user="Say: pong",
                max_output_tokens=256,
            )

            cleaned = output.strip()

            if not cleaned:
                raise RuntimeError(
                    "provider returned empty visible text"
                )

            print(
                f"OK   text[{tier.value}] "
                f"-> {cleaned[:80]!r}"
            )

        except Exception as exc:
            print(
                f"FAIL text[{tier.value}]: "
                f"{type(exc).__name__}: {exc}"
            )
            status = 1

    try:
        result = await client.generate_structured(
            purpose="smoke",
            tier=Tier.FAST,
            system="Fill the schema exactly.",
            user=(
                "Set ok to true and word to "
                "the string 'pong'."
            ),
            schema=Ping,
        )

        print(
            f"OK   structured -> {result}"
        )

    except Exception as exc:
        print(
            "FAIL structured: "
            f"{type(exc).__name__}: {exc}"
        )
        traceback.print_exc(limit=3)
        status = 1

    return status


if __name__ == "__main__":
    sys.exit(
        asyncio.run(main())
    )