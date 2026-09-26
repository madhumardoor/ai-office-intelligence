import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    async_sessionmaker,
    create_async_engine,
)

from app.llm.client import LLMClient
from app.llm.mock import MockLLM
from app.llm.pricing import PriceTable
from app.llm.types import Tier
from app.llm.usage import UsageRecorder


pytestmark = pytest.mark.integration


async def test_usage_persisted_without_prompt_text(
    migrated_db,
) -> None:
    engine = create_async_engine(
        str(migrated_db["url"])
    )

    session_factory = async_sessionmaker(
        engine,
        expire_on_commit=False,
    )

    async with engine.begin() as connection:
        await connection.execute(
            text("TRUNCATE llm_logs")
        )

    prices = PriceTable.from_json(
        '{"mock-fast": {'
        '"input_per_million": 1.0, '
        '"output_per_million": 2.0'
        '}}'
    )

    client = LLMClient(
        MockLLM(),
        prices=prices,
        recorder=UsageRecorder(
            session_factory
        ),
    )

    await client.generate_text(
        purpose="routing",
        tier=Tier.FAST,
        system="SECRET SYSTEM",
        user="PRIVATE QUESTION",
    )

    async with session_factory() as session:
        row = (
            await session.execute(
                text(
                    """
                    SELECT
                        provider,
                        model,
                        purpose,
                        prompt_hash,
                        input_tokens,
                        estimated_cost_usd
                    FROM llm_logs
                    """
                )
            )
        ).one()

        dump = str(
            (
                await session.execute(
                    text("SELECT * FROM llm_logs")
                )
            ).all()
        )

    await engine.dispose()

    assert row[:3] == (
        "mock",
        "mock-fast",
        "routing",
    )

    assert len(row[3]) == 64
    assert row[4] > 0
    assert row[5] is not None

    # Privacy guarantee:
    # actual prompt text must never be persisted.
    assert "PRIVATE QUESTION" not in dump
    assert "SECRET SYSTEM" not in dump


async def test_logging_failure_never_breaks_the_call(
    migrated_db,
) -> None:
    class BrokenFactory:
        def __call__(self):  # type: ignore[no-untyped-def]
            raise RuntimeError("db down")

    client = LLMClient(
        MockLLM(),
        recorder=UsageRecorder(
            BrokenFactory()
        ),  # type: ignore[arg-type]
    )

    result = await client.generate_text(
        purpose="test",
        tier=Tier.FAST,
        system="s",
        user="u",
    )

    assert result == "mock answer"