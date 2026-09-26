"""Company domain: canonical entity, aliases, identifiers, locations, and time-series facts."""

from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Any

from geoalchemy2 import Geography
from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base
from app.models.enums import MatchStatus, check_in
from app.models.mixins import ProvenanceMixin, TimestampMixin, UUIDPrimaryKeyMixin


class Company(UUIDPrimaryKeyMixin, TimestampMixin, ProvenanceMixin, Base):
    """Canonical company record (one row per real-world company after entity resolution)."""

    __tablename__ = "companies"
    __table_args__ = (
        CheckConstraint("employee_count IS NULL OR employee_count >= 0", name="employee_count_nonneg"),
        # Domain is the strongest identity signal; unique among live (non-deleted) rows.
        Index(
            "uq_companies_domain_live",
            "domain",
            unique=True,
            postgresql_where=text("domain IS NOT NULL AND deleted_at IS NULL"),
        ),
        Index(
            "ix_companies_name_trgm",
            "normalized_name",
            postgresql_using="gin",
            postgresql_ops={"normalized_name": "gin_trgm_ops"},
        ),
        Index("ix_companies_city_industry", "city", "industry"),
    )

    name: Mapped[str] = mapped_column(String(300), nullable=False)
    # Lowercased, legal-suffix-stripped name used for matching (Pvt Ltd, Inc, LLP ...).
    normalized_name: Mapped[str] = mapped_column(String(300), nullable=False)
    website: Mapped[str | None] = mapped_column(Text)
    domain: Mapped[str | None] = mapped_column(String(255))
    linkedin_url: Mapped[str | None] = mapped_column(Text)
    industry: Mapped[str | None] = mapped_column(String(150))
    description: Mapped[str | None] = mapped_column(Text)
    # Latest known headcount snapshot (history lives in company_employees).
    employee_count: Mapped[int | None] = mapped_column(Integer)
    city: Mapped[str | None] = mapped_column(String(100), index=True)
    state: Mapped[str | None] = mapped_column(String(100))
    country: Mapped[str] = mapped_column(String(2), nullable=False, default="IN", server_default="IN")
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    merged_into_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("companies.id", ondelete="SET NULL")
    )
    raw_payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict, server_default="{}")

    aliases: Mapped[list[CompanyAlias]] = relationship(back_populates="company", cascade="all, delete-orphan")
    locations: Mapped[list[CompanyLocation]] = relationship(back_populates="company", cascade="all, delete-orphan")


class CompanyAlias(UUIDPrimaryKeyMixin, TimestampMixin, ProvenanceMixin, Base):
    """Alternate names ("ABC Tech" for "ABC Technologies Pvt Ltd"). Supports entity resolution."""

    __tablename__ = "company_aliases"
    __table_args__ = (
        UniqueConstraint("company_id", "normalized_alias", name="uq_company_aliases_company_alias"),
        Index(
            "ix_company_aliases_trgm",
            "normalized_alias",
            postgresql_using="gin",
            postgresql_ops={"normalized_alias": "gin_trgm_ops"},
        ),
    )

    company_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("companies.id", ondelete="CASCADE"), nullable=False, index=True
    )
    alias: Mapped[str] = mapped_column(String(300), nullable=False)
    normalized_alias: Mapped[str] = mapped_column(String(300), nullable=False)
    company: Mapped[Company] = relationship(back_populates="aliases")


class CompanyIdentifier(UUIDPrimaryKeyMixin, TimestampMixin, ProvenanceMixin, Base):
    """External identifiers (domain, LinkedIn URL, CIN, GSTIN...). Each (type, value) is globally unique."""

    __tablename__ = "company_identifiers"
    __table_args__ = (UniqueConstraint("id_type", "id_value", name="uq_company_identifiers_type_value"),)

    company_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("companies.id", ondelete="CASCADE"), nullable=False, index=True
    )
    id_type: Mapped[str] = mapped_column(String(30), nullable=False)
    id_value: Mapped[str] = mapped_column(String(300), nullable=False)


class CompanyLocation(UUIDPrimaryKeyMixin, TimestampMixin, ProvenanceMixin, Base):
    """A company office/site with PostGIS geography (meters-accurate distance queries)."""

    __tablename__ = "company_locations"
    __table_args__ = (
        CheckConstraint("latitude IS NULL OR latitude BETWEEN -90 AND 90", name="lat_range"),
        CheckConstraint("longitude IS NULL OR longitude BETWEEN -180 AND 180", name="lng_range"),
        Index("ix_company_locations_geog", "geog", postgresql_using="gist"),
    )

    company_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("companies.id", ondelete="CASCADE"), nullable=False, index=True
    )
    label: Mapped[str | None] = mapped_column(String(150))  # e.g. "HQ", "Bangalore office"
    address: Mapped[str | None] = mapped_column(Text)
    area: Mapped[str | None] = mapped_column(String(120), index=True)  # e.g. "Whitefield"
    city: Mapped[str | None] = mapped_column(String(100), index=True)
    latitude: Mapped[float | None] = mapped_column(Float)
    longitude: Mapped[float | None] = mapped_column(Float)
    geog = mapped_column(Geography(geometry_type="POINT", srid=4326, spatial_index=False))
    is_primary: Mapped[bool] = mapped_column(nullable=False, default=False, server_default="false")
    company: Mapped[Company] = relationship(back_populates="locations")


class CompanyEmployeeSnapshot(UUIDPrimaryKeyMixin, TimestampMixin, ProvenanceMixin, Base):
    """Append-only headcount observations. Growth = delta between snapshots."""

    __tablename__ = "company_employees"
    __table_args__ = (
        CheckConstraint("employee_count >= 0", name="employee_count_nonneg"),
        UniqueConstraint("company_id", "observed_on", "source_id", name="uq_company_employees_obs"),
        Index("ix_company_employees_company_date", "company_id", "observed_on"),
    )

    company_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("companies.id", ondelete="CASCADE"), nullable=False
    )
    observed_on: Mapped[date] = mapped_column(Date, nullable=False)
    employee_count: Mapped[int] = mapped_column(Integer, nullable=False)
    location_scope: Mapped[str | None] = mapped_column(String(100))  # e.g. "Bangalore" or NULL = total


class CompanyHiringSnapshot(UUIDPrimaryKeyMixin, TimestampMixin, ProvenanceMixin, Base):
    """Append-only open-role counts (from careers pages / job APIs)."""

    __tablename__ = "company_hiring_signals"
    __table_args__ = (
        CheckConstraint("open_roles >= 0", name="open_roles_nonneg"),
        UniqueConstraint("company_id", "observed_on", "city", "source_id", name="uq_company_hiring_obs"),
        Index("ix_company_hiring_company_date", "company_id", "observed_on"),
    )

    company_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("companies.id", ondelete="CASCADE"), nullable=False
    )
    observed_on: Mapped[date] = mapped_column(Date, nullable=False)
    city: Mapped[str] = mapped_column(String(100), nullable=False, default="", server_default="")
    open_roles: Mapped[int] = mapped_column(Integer, nullable=False)
    departments: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict, server_default="{}")
    leadership_roles: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    evidence_url: Mapped[str | None] = mapped_column(Text)


class CompanyFunding(UUIDPrimaryKeyMixin, TimestampMixin, ProvenanceMixin, Base):
    """Funding events. Amount stored in minor-agnostic USD to allow comparison; original text preserved."""

    __tablename__ = "company_funding"
    __table_args__ = (
        CheckConstraint("amount_usd IS NULL OR amount_usd >= 0", name="amount_nonneg"),
        UniqueConstraint("company_id", "announced_on", "round_name", name="uq_company_funding_event"),
    )

    company_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("companies.id", ondelete="CASCADE"), nullable=False, index=True
    )
    announced_on: Mapped[date] = mapped_column(Date, nullable=False)
    round_name: Mapped[str] = mapped_column(String(80), nullable=False)
    amount_usd: Mapped[int | None] = mapped_column(BigInteger)
    investors: Mapped[str | None] = mapped_column(Text)
    evidence_url: Mapped[str | None] = mapped_column(Text)


class CompanyNews(UUIDPrimaryKeyMixin, TimestampMixin, ProvenanceMixin, Base):
    """News items tied to a company. `url_hash` prevents duplicate inserts across refresh runs."""

    __tablename__ = "company_news"
    __table_args__ = (
        UniqueConstraint("company_id", "url_hash", name="uq_company_news_company_url"),
        Index("ix_company_news_company_published", "company_id", "published_at"),
    )

    company_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("companies.id", ondelete="CASCADE"), nullable=False
    )
    title: Mapped[str] = mapped_column(Text, nullable=False)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    url_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    snippet: Mapped[str | None] = mapped_column(Text)
    category: Mapped[str | None] = mapped_column(String(60))  # expansion, funding, hiring...


class EntityMatchCandidate(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A proposed duplicate pair with per-feature scores. Never auto-merge without a stored confidence."""

    __tablename__ = "entity_match_candidates"
    __table_args__ = (
        CheckConstraint("confidence BETWEEN 0 AND 1", name="confidence_range"),
        CheckConstraint(check_in("status", MatchStatus), name="status_valid"),
        CheckConstraint("company_a_id <> company_b_id", name="distinct_pair"),
        UniqueConstraint("company_a_id", "company_b_id", name="uq_entity_match_pair"),
    )

    company_a_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("companies.id", ondelete="CASCADE"), nullable=False, index=True
    )
    company_b_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("companies.id", ondelete="CASCADE"), nullable=False, index=True
    )
    confidence: Mapped[float] = mapped_column(nullable=False)
    features: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict, server_default="{}")
    status: Mapped[str] = mapped_column(String(20), nullable=False, default=MatchStatus.NEEDS_REVIEW)
    reviewed_by: Mapped[str | None] = mapped_column(String(120))
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))