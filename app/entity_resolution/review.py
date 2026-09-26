"""Review-queue operations for ambiguous entity matches."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.entity_resolution.similarity import MatchScore
from app.models import EntityMatchCandidate
from app.models.enums import MatchStatus


def _ordered(a: uuid.UUID, b: uuid.UUID) -> tuple[uuid.UUID, uuid.UUID]:
    """Canonical (a < b) ordering so the unique pair constraint dedupes A/B vs B/A."""
    return (a, b) if str(a) < str(b) else (b, a)


async def enqueue_review(
    session: AsyncSession, company_a: uuid.UUID, company_b: uuid.UUID, score: MatchScore
) -> None:
    """Idempotent: re-ingesting the same file does not create duplicate review items,
    and never overwrites a human decision (only refreshes still-pending items)."""
    a, b = _ordered(company_a, company_b)
    stmt = (
        insert(EntityMatchCandidate)
        .values(
            company_a_id=a,
            company_b_id=b,
            confidence=score.confidence,
            features=score.features,
            status=MatchStatus.NEEDS_REVIEW,
        )
        .on_conflict_do_update(
            constraint="uq_entity_match_pair",
            set_={"confidence": score.confidence, "features": score.features},
            where=EntityMatchCandidate.status == MatchStatus.NEEDS_REVIEW,
        )
    )
    await session.execute(stmt)


async def list_pending(session: AsyncSession, limit: int = 50) -> list[EntityMatchCandidate]:
    stmt = (
        select(EntityMatchCandidate)
        .where(EntityMatchCandidate.status == MatchStatus.NEEDS_REVIEW)
        .order_by(EntityMatchCandidate.confidence.desc())
        .limit(limit)
    )
    return list((await session.execute(stmt)).scalars())


async def resolve_review(
    session: AsyncSession, candidate_id: uuid.UUID, *, confirm: bool, reviewer: str
) -> EntityMatchCandidate:
    """Record a human decision. This records the decision only; performing the actual merge
    (re-pointing child rows) is a separate, deliberate admin workflow."""
    cand = (
        await session.execute(select(EntityMatchCandidate).where(EntityMatchCandidate.id == candidate_id))
    ).scalar_one()
    cand.status = MatchStatus.CONFIRMED if confirm else MatchStatus.REJECTED
    cand.reviewed_by = reviewer
    cand.reviewed_at = datetime.now(UTC)
    await session.flush()
    return cand