"""Coworking domain: operators, branches, buildings, tenants, and properties."""

from __future__ import annotations

import uuid
from datetime import date

from geoalchemy2 import Geography
from sqlalchemy import (
    CheckConstraint,
    Date,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base
from app.models.mixins import ProvenanceMixin, TimestampMixin, UUIDPrimaryKeyMixin


class CoworkingOperator(UUIDPrimaryKeyMixin, TimestampMixin, ProvenanceMixin, Base):
    """A coworking brand. One operator has many branches."""

    __tablename__ = "coworking_operators"
    __table_args__ = (UniqueConstraint("name", name="uq_coworking_operators_name"),)

    name: Mapped[str] = mapped_column(String(200), nullable=False)
    website: Mapped[str | None] = mapped_column(Text)


class CoworkingBuilding(UUIDPrimaryKeyMixin, TimestampMixin, ProvenanceMixin, Base):
    """A physical commercial building; can host several operators' branches."""

    __tablename__ = "coworking_buildings"
    __table_args__ = (
        CheckConstraint("latitude BETWEEN -90 AND 90", name="lat_range"),
        CheckConstraint("longitude BETWEEN -180 AND 180", name="lng_range"),
        UniqueConstraint("name", "city", name="uq_coworking_buildings_name_city"),
        Index("ix_coworking_buildings_geog", "geog", postgresql_using="gist"),
    )

    name: Mapped[str] = mapped_column(String(250), nullable=False)
    address: Mapped[str | None] = mapped_column(Text)
    area: Mapped[str | None] = mapped_column(String(120), index=True)
    city: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)
    geog = mapped_column(Geography(geometry_type="POINT", srid=4326, spatial_index=False), nullable=False)
    total_floors: Mapped[int | None] = mapped_column(Integer)


class CoworkingBranch(UUIDPrimaryKeyMixin, TimestampMixin, ProvenanceMixin, Base):
    """An operator's centre inside a building."""

    __tablename__ = "coworking_branches"
    __table_args__ = (
        CheckConstraint("seat_capacity IS NULL OR seat_capacity > 0", name="capacity_positive"),
        UniqueConstraint("operator_id", "building_id", "name", name="uq_coworking_branches_identity"),
    )

    operator_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("coworking_operators.id", ondelete="CASCADE"), nullable=False, index=True
    )
    building_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("coworking_buildings.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(250), nullable=False)
    seat_capacity: Mapped[int | None] = mapped_column(Integer)
    floor: Mapped[str | None] = mapped_column(String(30))


class CoworkingTenant(UUIDPrimaryKeyMixin, TimestampMixin, ProvenanceMixin, Base):
    """Company <-> branch tenancy over time. last_seen_on enables new/departed tenant detection."""

    __tablename__ = "coworking_tenants"
    __table_args__ = (
        CheckConstraint("seats_used IS NULL OR seats_used > 0", name="seats_positive"),
        CheckConstraint("last_seen_on >= first_seen_on", name="seen_order"),
        UniqueConstraint("company_id", "branch_id", name="uq_coworking_tenants_company_branch"),
        Index("ix_coworking_tenants_branch", "branch_id"),
    )

    company_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("companies.id", ondelete="CASCADE"), nullable=False, index=True
    )
    branch_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("coworking_branches.id", ondelete="CASCADE"), nullable=False
    )
    seats_used: Mapped[int | None] = mapped_column(Integer)
    first_seen_on: Mapped[date] = mapped_column(Date, nullable=False)
    last_seen_on: Mapped[date] = mapped_column(Date, nullable=False)
    # PII stays on the base table only; the agent-facing view excludes these columns.
    contact_email: Mapped[str | None] = mapped_column(String(320))
    contact_phone: Mapped[str | None] = mapped_column(String(40))


class Property(UUIDPrimaryKeyMixin, TimestampMixin, ProvenanceMixin, Base):
    """Commercial properties (non-coworking) for proximity analysis."""

    __tablename__ = "properties"
    __table_args__ = (
        CheckConstraint("latitude BETWEEN -90 AND 90", name="lat_range"),
        CheckConstraint("longitude BETWEEN -180 AND 180", name="lng_range"),
        Index("ix_properties_geog", "geog", postgresql_using="gist"),
    )

    name: Mapped[str] = mapped_column(String(250), nullable=False)
    property_type: Mapped[str] = mapped_column(String(60), nullable=False)
    area: Mapped[str | None] = mapped_column(String(120), index=True)
    city: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)
    geog = mapped_column(Geography(geometry_type="POINT", srid=4326, spatial_index=False), nullable=False)
    total_area_sqft: Mapped[int | None] = mapped_column(Integer)
    description: Mapped[str | None] = mapped_column(Text)