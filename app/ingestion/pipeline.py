"""Ingestion orchestration.

Guarantees:
  * Each row runs in its own transaction: one bad row never poisons the batch.
  * Re-running the same file is idempotent (natural-key matching; upserts).
  * Exact domain / LinkedIn matches are resolved before fuzzy entity resolution.
  * Progress is checkpointed every N rows into ingestion_runs; a crashed run resumes
    after the last committed row.
  * Nothing is silently dropped: failures land in the report.
  * A canonical company may have multiple physical CompanyLocation records.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import UTC, date, datetime
from pathlib import Path

from geoalchemy2.elements import WKTElement
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import Settings
from app.entity_resolution.resolver import Decision, EntityResolver
from app.entity_resolution.review import enqueue_review
from app.entity_resolution.similarity import Candidate
from app.ingestion.geocoder import Geocoder
from app.ingestion.loaders import IngestionFileError, iter_from, load_file
from app.ingestion.report import IngestionReport, RowIssue
from app.ingestion.schemas import CompanyRow
from app.models import (
    Company,
    CompanyAlias,
    CompanyIdentifier,
    CompanyLocation,
    CoworkingBranch,
    CoworkingBuilding,
    CoworkingOperator,
    CoworkingTenant,
    IngestionRun,
    Source,
)
from app.models.enums import IngestionStatus, SourceType

logger = logging.getLogger(__name__)


_MERGEABLE = (
    "industry",
    "employee_count",
    "state",
    "linkedin_url",
    "domain",
    "city",
)


def _file_hash(path: Path) -> str:
    """Return SHA-256 hash of a file."""
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)

    return h.hexdigest()


class _RowRejected(Exception):
    """Expected row-level rejection."""

    def __init__(
        self,
        message: str,
        kind: str,
        name: str | None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.name = name


class IngestionPipeline:
    """Orchestrates file ingestion into the database."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        settings: Settings,
        geocoder: Geocoder | None = None,
    ) -> None:
        self._sf = session_factory
        self._settings = settings
        self._geocoder = geocoder

        self._resolver = EntityResolver(
            settings.er_auto_match_threshold,
            settings.er_review_threshold,
        )

    async def ingest_file(
        self,
        path: str | Path,
        *,
        resume_run_id: uuid.UUID | None = None,
    ) -> IngestionReport:
        """Ingest a CSV/Excel file."""

        p = Path(path)
        report = IngestionReport(file_name=p.name)

        # Structural validation happens before any database write.
        loaded = load_file(p)

        report.unknown_columns = loaded.unknown_columns
        report.rows_total = len(loaded.rows)

        file_hash = _file_hash(p)

        async with self._sf() as session:
            source = await self._get_or_create_source(
                session,
                p,
                file_hash,
            )

            run, start_after = await self._start_run(
                session,
                p,
                source.id,
                resume_run_id,
                report,
            )

            await session.commit()

            run_id = run.id
            source_id = source.id

            report.resumed_from_row = start_after

        processed_since_checkpoint = 0
        last_row = start_after

        for row_number, raw in iter_from(loaded, start_after):
            async with self._sf() as session:
                try:
                    async with session.begin():
                        await self._process_row(
                            session,
                            row_number,
                            raw,
                            source_id,
                            report,
                        )

                except _RowRejected as rej:
                    report.failed += 1

                    report.add_issue(
                        RowIssue(
                            row_number,
                            rej.name,
                            rej.kind,
                            str(rej),
                        )
                    )

                except Exception as exc:  # noqa: BLE001
                    # Each row has its own transaction.
                    # A failure here must not poison the remaining rows.
                    report.failed += 1

                    report.add_issue(
                        RowIssue(
                            row_number,
                            None,
                            "db_error",
                            f"{type(exc).__name__}: {exc}"[:300],
                        )
                    )

                    logger.warning(
                        "row failed",
                        extra={
                            "row": row_number,
                            "error": type(exc).__name__,
                        },
                    )

            last_row = row_number
            processed_since_checkpoint += 1

            if (
                processed_since_checkpoint
                >= self._settings.ingest_checkpoint_every
            ):
                await self._checkpoint(
                    run_id,
                    last_row,
                    report,
                )

                processed_since_checkpoint = 0

        await self._finish(
            run_id,
            last_row,
            report,
        )

        return report

    # ------------------------------------------------------------------
    # Run bookkeeping
    # ------------------------------------------------------------------

    async def _get_or_create_source(
        self,
        session: AsyncSession,
        p: Path,
        file_hash: str,
    ) -> Source:
        """Create or update the ingestion source."""

        stype = (
            SourceType.CSV
            if p.suffix.lower() == ".csv"
            else SourceType.EXCEL
        )

        stmt = (
            insert(Source)
            .values(
                source_type=stype,
                name=p.name,
                reliability=0.6,
                extra={"sha256": file_hash},
            )
            .on_conflict_do_update(
                constraint="uq_sources_type_name",
                set_={"extra": {"sha256": file_hash}},
            )
        )

        await session.execute(stmt)

        return (
            await session.execute(
                select(Source).where(
                    Source.source_type == stype,
                    Source.name == p.name,
                )
            )
        ).scalar_one()

    async def _start_run(
        self,
        session: AsyncSession,
        p: Path,
        source_id: uuid.UUID,
        resume_run_id: uuid.UUID | None,
        report: IngestionReport,
    ) -> tuple[IngestionRun, int]:
        """Create a new ingestion run or resume an existing one."""

        if resume_run_id:
            run = (
                await session.execute(
                    select(IngestionRun).where(
                        IngestionRun.id == resume_run_id
                    )
                )
            ).scalar_one()

            if run.status == IngestionStatus.SUCCEEDED:
                raise IngestionFileError(
                    "run already succeeded; nothing to resume"
                )

            saved = run.report or {}

            for key in (
                "inserted",
                "updated",
                "unchanged",
                "needs_review",
                "failed",
                "geocoded",
                "geocode_missing",
                "tenancies_upserted",
            ):
                setattr(
                    report,
                    key,
                    int(saved.get(key, 0)),
                )

            run.status = IngestionStatus.RUNNING

            return run, int(
                (run.checkpoint or {}).get(
                    "last_row",
                    0,
                )
            )

        run = IngestionRun(
            job_name=f"ingest:{p.name}",
            status=IngestionStatus.RUNNING,
            started_at=datetime.now(UTC),
            rows_total=report.rows_total,
            source_id=source_id,
        )

        session.add(run)
        await session.flush()

        return run, 0

    async def _checkpoint(
        self,
        run_id: uuid.UUID,
        last_row: int,
        report: IngestionReport,
    ) -> None:
        """Persist ingestion checkpoint and current report."""

        async with self._sf() as session, session.begin():
            run = (
                await session.execute(
                    select(IngestionRun).where(
                        IngestionRun.id == run_id
                    )
                )
            ).scalar_one()

            run.checkpoint = {
                "last_row": last_row,
            }

            run.report = report.to_dict()

    async def _finish(
        self,
        run_id: uuid.UUID,
        last_row: int,
        report: IngestionReport,
    ) -> None:
        """Mark ingestion run complete."""

        async with self._sf() as session, session.begin():
            run = (
                await session.execute(
                    select(IngestionRun).where(
                        IngestionRun.id == run_id
                    )
                )
            ).scalar_one()

            run.status = report.status
            run.finished_at = datetime.now(UTC)

            run.checkpoint = {
                "last_row": last_row,
            }

            run.rows_inserted = report.inserted
            run.rows_updated = report.updated
            run.rows_skipped = report.unchanged
            run.rows_failed = report.failed

            run.report = report.to_dict()

            if report.failed:
                run.error_summary = (
                    f"{report.failed} row(s) failed; "
                    "see report.issues"
                )

    # ------------------------------------------------------------------
    # Per-row processing
    # ------------------------------------------------------------------

    async def _process_row(
        self,
        session: AsyncSession,
        row_number: int,
        raw: dict[str, object],
        source_id: uuid.UUID,
        report: IngestionReport,
    ) -> None:
        """Validate, resolve, upsert, and enrich one row."""

        # --------------------------------------------------------------
        # 1. Validate and normalize incoming row
        # --------------------------------------------------------------

        try:
            row = CompanyRow.from_raw(
                row_number,
                raw,
                self._settings.default_country_code,
            )

        except (ValueError, ValidationError) as exc:
            if isinstance(exc, ValidationError):
                first = exc.errors()[0]

                detail = (
                    f"{'.'.join(str(x) for x in first['loc'])}: "
                    f"{first['msg']}"
                )
            else:
                detail = str(exc)

            raise _RowRejected(
                detail[:200],
                "validation_error",
                None,
            ) from exc

        now = datetime.now(UTC)

        # --------------------------------------------------------------
        # 2. EXACT NATURAL-KEY LOOKUP
        #
        # Domain and LinkedIn are company-level identity signals.
        # We use them to find the canonical Company.
        #
        # Physical location is handled separately by _upsert_location().
        # --------------------------------------------------------------

        company: Company | None = None

        # 2A. Exact domain match
        if row.domain:
            company = (
                await session.execute(
                    select(Company).where(
                        Company.domain == row.domain,
                        Company.deleted_at.is_(None),
                    )
                )
            ).scalar_one_or_none()

        # 2B. Exact LinkedIn match
        if company is None and row.linkedin_url:
            company = (
                await session.execute(
                    select(Company).where(
                        Company.linkedin_url == row.linkedin_url,
                        Company.deleted_at.is_(None),
                    )
                )
            ).scalar_one_or_none()

        # --------------------------------------------------------------
        # 3. EXISTING COMPANY FOUND BY EXACT IDENTIFIER
        # --------------------------------------------------------------

        if company is not None:
            changed = self._merge_fields(
                company,
                row,
            )

            if changed:
                report.updated += 1
            else:
                report.unchanged += 1

        # --------------------------------------------------------------
        # 4. NO EXACT MATCH -> ENTITY RESOLUTION
        # --------------------------------------------------------------

        else:
            candidate = Candidate(
                normalized_name=row.normalized_name,
                domain=row.domain,
                linkedin_url=row.linkedin_url,
                city=row.city,
                address=row.address,
                latitude=row.latitude,
                longitude=row.longitude,
                operator_name=row.coworking_operator,
            )

            resolution = await self._resolver.resolve(
                session,
                candidate,
            )

            # ----------------------------------------------------------
            # 4A. Resolver found an automatic match
            # ----------------------------------------------------------

            if (
                resolution.decision == Decision.AUTO_MATCH
                and resolution.company_id
            ):
                company = (
                    await session.execute(
                        select(Company).where(
                            Company.id == resolution.company_id,
                            Company.deleted_at.is_(None),
                        )
                    )
                ).scalar_one()

                changed = self._merge_fields(
                    company,
                    row,
                )

                if changed:
                    report.updated += 1
                else:
                    report.unchanged += 1

            # ----------------------------------------------------------
            # 4B. No automatic match -> insert new company
            # ----------------------------------------------------------

            else:
                company = Company(
                    name=row.company_name,
                    normalized_name=row.normalized_name,
                    website=row.website,
                    domain=row.domain,
                    linkedin_url=row.linkedin_url,
                    industry=row.industry,
                    employee_count=row.employee_count,
                    city=row.city,
                    state=row.state,
                    source_id=source_id,
                    retrieved_at=now,
                    raw_payload={
                        "row_number": row_number,
                        "input": {
                            k: str(v)
                            for k, v in raw.items()
                        },
                    },
                )

                session.add(company)
                await session.flush()

                report.inserted += 1

                # ------------------------------------------------------
                # 4C. Needs-review resolution
                # ------------------------------------------------------

                if (
                    resolution.decision == Decision.NEEDS_REVIEW
                    and resolution.score
                ):
                    for (
                        other_id,
                        score,
                    ) in resolution.review_candidates:
                        await enqueue_review(
                            session,
                            company.id,
                            other_id,
                            score,
                        )

                    report.needs_review += 1

                    report.add_issue(
                        RowIssue(
                            row_number,
                            row.company_name,
                            "needs_review",
                            (
                                "possible duplicate "
                                f"(confidence "
                                f"{resolution.score.confidence:.2f})"
                            ),
                        )
                    )

        # --------------------------------------------------------------
        # 5. Provenance / aliases / identifiers
        # --------------------------------------------------------------

        await self._upsert_alias(
            session,
            company.id,
            row,
            source_id,
        )

        await self._upsert_identifiers(
            session,
            company.id,
            row,
            source_id,
        )

        # --------------------------------------------------------------
        # 6. Physical company location
        # --------------------------------------------------------------

        await self._upsert_location(
            session,
            company.id,
            row,
            source_id,
            now,
            report,
        )

        # --------------------------------------------------------------
        # 7. Coworking tenancy
        # --------------------------------------------------------------

        await self._upsert_tenancy(
            session,
            company.id,
            row,
            source_id,
            now,
            report,
        )

    # ------------------------------------------------------------------
    # Company merging
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_fields(
        company: Company,
        row: CompanyRow,
    ) -> bool:
        """Merge newer non-null company-level values.

        Website is intentionally excluded because coworking source rows
        frequently contain location-specific website URLs. The canonical
        company website must not be overwritten by a branch/location URL.

        Existing values are never replaced by None.
        """

        changed = False

        for field_name in _MERGEABLE:
            new_value = getattr(row, field_name)

            if (
                new_value is not None
                and getattr(company, field_name) != new_value
            ):
                setattr(company, field_name, new_value)
                changed = True

        return changed

    # ------------------------------------------------------------------
    # Alias upsert
    # ------------------------------------------------------------------

    @staticmethod
    async def _upsert_alias(
        session: AsyncSession,
        company_id: uuid.UUID,
        row: CompanyRow,
        source_id: uuid.UUID,
    ) -> None:
        """Create company alias if it does not already exist."""

        await session.execute(
            insert(CompanyAlias)
            .values(
                company_id=company_id,
                alias=row.company_name,
                normalized_alias=row.normalized_name,
                source_id=source_id,
            )
            .on_conflict_do_nothing(
                constraint="uq_company_aliases_company_alias"
            )
        )

    # ------------------------------------------------------------------
    # Identifier upsert
    # ------------------------------------------------------------------

    @staticmethod
    async def _upsert_identifiers(
        session: AsyncSession,
        company_id: uuid.UUID,
        row: CompanyRow,
        source_id: uuid.UUID,
    ) -> None:
        """Upsert domain and LinkedIn identifiers."""

        for id_type, value in (
            ("domain", row.domain),
            ("linkedin", row.linkedin_url),
        ):
            if value:
                await session.execute(
                    insert(CompanyIdentifier)
                    .values(
                        company_id=company_id,
                        id_type=id_type,
                        id_value=value,
                        source_id=source_id,
                    )
                    .on_conflict_do_nothing(
                        constraint="uq_company_identifiers_type_value"
                    )
                )

    # ------------------------------------------------------------------
    # Location upsert
    # ------------------------------------------------------------------

    async def _upsert_location(
        self,
        session: AsyncSession,
        company_id: uuid.UUID,
        row: CompanyRow,
        source_id: uuid.UUID,
        now: datetime,
        report: IngestionReport,
    ) -> None:
        """Insert/update a physical company location.

        A single Company can have multiple physical locations.

        Location identity uses:
          1. same normalized physical address
          2. same label + same area + nearby coordinates
          3. same label + nearby coordinates
          4. same label + same area when coordinates are unavailable

        A canonical company can therefore have multiple branches.
        Different addresses create separate CompanyLocation records.
        """

        if not any(
            (
                row.address,
                row.area,
                row.latitude is not None,
                row.longitude is not None,
                row.city,
            )
        ):
            return

        lat = row.latitude
        lng = row.longitude

        # --------------------------------------------------------------
        # Optional geocoding
        # --------------------------------------------------------------

        if (
            lat is None
            and self._geocoder
            and (row.address or row.area)
        ):
            query = ", ".join(
                x
                for x in (
                    row.address,
                    row.area,
                    row.city,
                    "India",
                )
                if x
            )

            geo = await self._geocoder.geocode(
                session,
                query,
            )

            if geo:
                lat = geo.latitude
                lng = geo.longitude
                report.geocoded += 1
            else:
                report.geocode_missing += 1

                report.add_issue(
                    RowIssue(
                        row.row_number,
                        row.company_name,
                        "geocode_miss",
                        (
                            "no coordinates found; "
                            "location stored without geometry"
                        ),
                    )
                )

        # --------------------------------------------------------------
        # Location label
        # --------------------------------------------------------------

        label = (
            row.building_name
            or row.branch_name
            or "Office"
        )

        # --------------------------------------------------------------
        # Search all existing locations for this company
        # --------------------------------------------------------------

        locations = (
            await session.execute(
                select(CompanyLocation).where(
                    CompanyLocation.company_id == company_id,
                )
            )
        ).scalars().all()

        existing: CompanyLocation | None = None

        # Coordinate fallback tolerance. This is only used when the
        # normalized physical address is not an exact match.
        COORD_TOLERANCE = 0.005

        def normalize_address(value: str | None) -> str:
            """Normalize an address for duplicate-location comparison.

            Punctuation, spaces and casing are ignored so values such as:
                "12, MG Road, Bengaluru"
                "12 MG Road Bengaluru"
            are treated as the same physical address.

            We deliberately do not use fuzzy address matching here because
            a coworking operator can have multiple branches with similar
            address text.
            """
            if not value:
                return ""

            return "".join(
                ch
                for ch in value.casefold()
                if ch.isalnum()
            )

        incoming_address = normalize_address(row.address)

        for location in locations:
            existing_address = normalize_address(location.address)

            same_address = (
                bool(incoming_address)
                and bool(existing_address)
                and incoming_address == existing_address
            )

            same_label = (
                location.label == label
            )

            same_area = (
                location.area == row.area
                if row.area is not None
                else True
            )

            coordinates_match = False

            if (
                lat is not None
                and lng is not None
                and location.latitude is not None
                and location.longitude is not None
            ):
                coordinates_match = (
                    abs(location.latitude - lat)
                    <= COORD_TOLERANCE
                    and abs(location.longitude - lng)
                    <= COORD_TOLERANCE
                )

            # RULE 1: same company + same physical address = duplicate
            # location. Branch/building label does not need to be identical.
            if same_address:
                existing = location
                break

            # RULE 2: same branch/building + same area + coordinates.
            if (
                same_label
                and same_area
                and coordinates_match
            ):
                existing = location
                break

            # RULE 3: same branch/building + same physical coordinates.
            if (
                same_label
                and coordinates_match
            ):
                existing = location
                break

            # RULE 4: without coordinates, retain conservative idempotency
            # using the same label/area/address information.
            if (
                lat is None
                and lng is None
                and same_label
                and same_area
                and (
                    not incoming_address
                    or not existing_address
                    or incoming_address == existing_address
                )
            ):
                existing = location
                break

        # --------------------------------------------------------------
        # PostGIS geometry
        # --------------------------------------------------------------

        geog = (
            WKTElement(
                f"POINT({lng} {lat})",
                srid=4326,
            )
            if lat is not None and lng is not None
            else None
        )

        # --------------------------------------------------------------
        # Update existing location
        # --------------------------------------------------------------

        if existing is not None:
            changed = False

            if (
                row.address
                and existing.address != row.address
            ):
                existing.address = row.address
                changed = True

            if (
                row.area
                and existing.area != row.area
            ):
                existing.area = row.area
                changed = True

            if (
                row.city
                and existing.city != row.city
            ):
                existing.city = row.city
                changed = True

            if lat is not None and lng is not None:
                if (
                    existing.latitude != lat
                    or existing.longitude != lng
                ):
                    existing.latitude = lat
                    existing.longitude = lng
                    existing.geog = geog
                    changed = True

                elif existing.geog is None:
                    existing.geog = geog
                    changed = True

            if changed:
                existing.source_id = source_id
                existing.retrieved_at = now

            return

        # --------------------------------------------------------------
        # Create a new physical location
        # --------------------------------------------------------------

        session.add(
            CompanyLocation(
                company_id=company_id,
                label=label,
                address=row.address,
                area=row.area,
                city=row.city,
                latitude=lat,
                longitude=lng,
                geog=geog,
                source_id=source_id,
                retrieved_at=now,
            )
        )

    # ------------------------------------------------------------------
    # Coworking tenancy upsert
    # ------------------------------------------------------------------

    @staticmethod
    async def _upsert_tenancy(
        session: AsyncSession,
        company_id: uuid.UUID,
        row: CompanyRow,
        source_id: uuid.UUID,
        now: datetime,
        report: IngestionReport,
    ) -> None:
        """Upsert coworking tenancy when sufficient building data exists."""

        if not (
            row.coworking_operator
            and row.branch_name
            and row.building_name
        ):
            return

        if (
            row.latitude is None
            or row.longitude is None
        ):
            report.add_issue(
                RowIssue(
                    row.row_number,
                    row.company_name,
                    "validation_error",
                    (
                        "tenancy skipped: "
                        "building coordinates required"
                    ),
                )
            )

            return

        city = row.city or "Unknown"

        # --------------------------------------------------------------
        # Coworking operator
        # --------------------------------------------------------------

        op = (
            await session.execute(
                select(CoworkingOperator).where(
                    CoworkingOperator.name
                    == row.coworking_operator
                )
            )
        ).scalar_one_or_none()

        if op is None:
            op = CoworkingOperator(
                name=row.coworking_operator,
                source_id=source_id,
                retrieved_at=now,
            )

            session.add(op)
            await session.flush()

        # --------------------------------------------------------------
        # Coworking building
        # --------------------------------------------------------------

        bld = (
            await session.execute(
                select(CoworkingBuilding).where(
                    CoworkingBuilding.name
                    == row.building_name,
                    CoworkingBuilding.city
                    == city,
                )
            )
        ).scalar_one_or_none()

        if bld is None:
            bld = CoworkingBuilding(
                name=row.building_name,
                area=row.area,
                city=city,
                latitude=row.latitude,
                longitude=row.longitude,
                geog=WKTElement(
                    f"POINT({row.longitude} {row.latitude})",
                    srid=4326,
                ),
                address=row.address,
                source_id=source_id,
                retrieved_at=now,
            )

            session.add(bld)
            await session.flush()

        # --------------------------------------------------------------
        # Coworking branch
        # --------------------------------------------------------------

        br = (
            await session.execute(
                select(CoworkingBranch).where(
                    CoworkingBranch.operator_id == op.id,
                    CoworkingBranch.building_id == bld.id,
                    CoworkingBranch.name == row.branch_name,
                )
            )
        ).scalar_one_or_none()

        if br is None:
            br = CoworkingBranch(
                operator_id=op.id,
                building_id=bld.id,
                name=row.branch_name,
                source_id=source_id,
                retrieved_at=now,
            )

            session.add(br)
            await session.flush()

        # --------------------------------------------------------------
        # Tenant
        # --------------------------------------------------------------

        today = date.today()

        await session.execute(
            insert(CoworkingTenant)
            .values(
                company_id=company_id,
                branch_id=br.id,
                seats_used=row.seats_used,
                first_seen_on=today,
                last_seen_on=today,
                contact_email=row.tenant_email,
                contact_phone=row.tenant_phone,
                source_id=source_id,
                retrieved_at=now,
            )
            .on_conflict_do_update(
                constraint="uq_coworking_tenants_company_branch",
                set_={
                    "last_seen_on": today,
                    "seats_used": row.seats_used,
                    "retrieved_at": now,
                },
            )
        )

        report.tenancies_upserted += 1 