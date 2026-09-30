from __future__ import annotations

import argparse
from pathlib import Path

from app.company.onboarding import CompanyOnboardingService
from app.config import get_settings
from app.database import create_engine, create_session_factory
from app.rag.embedding.factory import create_embedder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create/find a company and ingest its website/documents into the existing RAG pipeline."
    )
    parser.add_argument("--name", required=True, help="Company name")
    parser.add_argument("--url", help="Optional company website URL")
    parser.add_argument(
        "--file",
        action="append",
        default=[],
        help="Optional PDF/TXT/HTML file. Repeat --file for multiple documents.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=12,
        help="Maximum same-domain website pages to crawl (default: 12)",
    )
    return parser


async def main() -> None:
    args = build_parser().parse_args()
    settings = get_settings()
    engine = create_engine(settings)
    session_factory = create_session_factory(engine)
    embedder = create_embedder(settings)

    service = CompanyOnboardingService(
        settings=settings,
        session_factory=session_factory,
        embedder=embedder,
        max_website_pages=args.max_pages,
    )

    try:
        result = await service.onboard(
            company_name=args.name,
            website=args.url,
            document_paths=[Path(p) for p in args.file],
        )
    finally:
        await engine.dispose()

    print("=" * 72)
    print("COMPANY ONBOARDING")
    print("=" * 72)
    print(f"Company       : {result.company_name}")
    print(f"Company ID    : {result.company_id}")
    print(f"Website       : {result.website or 'None'}")
    print(f"Pages crawled : {result.pages_discovered}")
    print(f"Docs created  : {result.documents_created}")
    print(f"Docs skipped  : {result.documents_skipped_duplicate}")
    print(f"Chunks created: {result.chunks_created}")
    print(f"Embeddings    : {result.embeddings_embedded}")
    print(f"Cache reused  : {result.embeddings_reused}")
    print(f"Pending embed : {result.embedding_pending}")
    print(f"Failures      : {len(result.failures)}")

    if result.failures:
        print("\nFailures:")
        for failure in result.failures[:50]:
            print(f" - {failure}")

    print("=" * 72)

    if result.embedding_pending or result.failures:
        raise SystemExit(2)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
