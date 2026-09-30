from __future__ import annotations

import argparse
import asyncio
import uuid

from sqlalchemy import func, select, update

from app.config import get_settings
from app.database import create_engine, create_session_factory
from app.models import Company, DocumentChunk
from app.rag.embedding.factory import create_embedder
from app.rag.embedding.service import EmbeddingService


async def main(company_name: str) -> None:
    settings = get_settings()

    engine = create_engine(settings)
    sf = create_session_factory(engine)

    try:
        async with sf() as session:
            result = await session.execute(
                select(Company.id, Company.name)
                .where(
                    func.lower(Company.name) == company_name.strip().lower(),
                    Company.deleted_at.is_(None),
                )
            )
            matches = result.all()

        if len(matches) != 1:
            raise SystemExit(
                f"Expected exactly one active company named {company_name!r}; "
                f"found {len(matches)}."
            )

        company_id: uuid.UUID = matches[0][0]
        resolved_name = matches[0][1]

        async with sf() as session, session.begin():
            reset = await session.execute(
                update(DocumentChunk)
                .where(DocumentChunk.company_id == company_id)
                .values(
                    embedding=None,
                    embedding_model=None,
                )
            )
            print(
                f"Reset embeddings for {reset.rowcount} chunk(s) "
                f"belonging to {resolved_name}."
            )

        embedder = create_embedder(settings)
        service = EmbeddingService(
            embedder,
            batch_size=settings.embedding_batch_size,
        )

        async with sf() as session, session.begin():
            # Only the selected company's chunks were reset. Other companies
            # with pending chunks will be processed too only if they already
            # had pending embeddings.
            stats = await service.embed_pending(session)

        print(f"Embedded: {stats.embedded}")
        print(f"Reused: {stats.reused_from_cache}")
        print(f"Pending after: {stats.pending_after}")
        print(f"Failed batches: {stats.failed_batches}")

    finally:
        await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--company-name",
        required=True,
        help="Exact active company name as stored in PostgreSQL.",
    )
    args = parser.parse_args()
    asyncio.run(main(args.company_name))
