"""CLI: uv run python -m scripts.ingest_file data/samples/companies_sample.csv [--resume RUN_ID]"""

from __future__ import annotations

import argparse
import asyncio
import json
import uuid

from app.config import get_settings
from app.config.logging import configure_logging
from app.database import create_engine, create_session_factory
from app.ingestion.geocoder import Geocoder
from app.ingestion.pipeline import IngestionPipeline


async def main(path: str, resume: str | None) -> None:
    settings = get_settings()
    configure_logging(settings.log_level, json_logs=False)
    engine = create_engine(settings)
    geocoder = Geocoder(settings) if settings.geocoding_enabled else None
    try:
        pipeline = IngestionPipeline(create_session_factory(engine), settings, geocoder)
        report = await pipeline.ingest_file(path, resume_run_id=uuid.UUID(resume) if resume else None)
        print(json.dumps(report.to_dict(), indent=2, default=str))
    finally:
        if geocoder:
            await geocoder.aclose()
        await engine.dispose()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--resume", help="ingestion_runs.id to resume")
    a = ap.parse_args()
    asyncio.run(main(a.path, a.resume))