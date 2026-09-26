"""CLI: index documents then embed them.

uv run python -m scripts.ingest_documents data/samples/docs --type news
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from app.config import get_settings
from app.config.logging import configure_logging
from app.database import create_engine, create_session_factory
from app.rag.embedding.factory import create_embedder
from app.rag.embedding.service import EmbeddingService
from app.rag.indexer import DocumentIndexer
from app.rag.loaders import SUPPORTED


async def main(target: str, doc_type: str, embed: bool) -> None:
    settings = get_settings()
    configure_logging(settings.log_level, json_logs=False)
    engine = create_engine(settings)
    sf = create_session_factory(engine)
    root = Path(target)
    files = sorted(p for p in ([root] if root.is_file() else root.rglob("*")) if p.suffix.lower() in SUPPORTED)
    indexer = DocumentIndexer(settings)
    created = skipped = chunks = 0
    failures: list[str] = []
    try:
        for f in files:
            async with sf() as session, session.begin():
                r = await indexer.index_path(session, f, document_type=doc_type)
            created += r.documents_created
            skipped += r.documents_skipped_duplicate
            chunks += r.chunks_created
            failures += r.failures
        print(f"files={len(files)} documents_created={created} duplicates_skipped={skipped} chunks={chunks}")
        for msg in failures:
            print(f"  FAILED: {msg}")
        if embed:
            svc = EmbeddingService(create_embedder(settings), settings.embedding_batch_size)
            async with sf() as session, session.begin():
                stats = await svc.embed_pending(session)
            print(
                f"embedded={stats.embedded} reused={stats.reused_from_cache} "
                f"pending={stats.pending_after} failed_batches={stats.failed_batches}"
            )
    finally:
        await engine.dispose()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--type", default="other")
    ap.add_argument("--no-embed", action="store_true", help="index only; embed later with backfill_embeddings")
    a = ap.parse_args()
    asyncio.run(main(a.path, a.type, not a.no_embed))