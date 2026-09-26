"""Input column contract, header aliases, and the validated row model."""

from __future__ import annotations

from pydantic import (
    BaseModel,
    ConfigDict,
    field_validator,
    model_validator,
)

from app.ingestion import normalizers as n


# ---------------------------------------------------------------------------
# Canonical column -> accepted header spellings.
# Headers are canonicalized by loaders.py before lookup.
# ---------------------------------------------------------------------------

COLUMN_ALIASES: dict[str, set[str]] = {
    "company_name": {
        "company_name",
        "company",
        "name",
        "organisation",
        "organization",
    },
    "website": {
        "website",
        "url",
        "company_website",
        "site",
    },
    "linkedin_url": {
        "linkedin_url",
        "linkedin",
        "linkedin_company_url",
    },
    "industry": {
        "industry",
        "sector",
    },
    "employee_count": {
        "employee_count",
        "employees",
        "headcount",
        "team_size",
    },
    "city": {
        "city",
    },
    "state": {
        "state",
    },
    "area": {
        "area",
        "locality",
        "micro_market",
    },
    "address": {
        "address",
        "office_address",
    },
    "latitude": {
        "latitude",
        "lat",
    },
    "longitude": {
        "longitude",
        "lng",
        "lon",
        "long",
    },
    "coworking_operator": {
        "coworking_operator",
        "operator",
    },
    "branch_name": {
        "branch_name",
        "branch",
        "centre",
        "center",
    },
    "building_name": {
        "building_name",
        "building",
    },
    "tenant_name": {
        "tenant_name",
        "tenant",
    },
    "tenant_website": {
        "tenant_website",
    },
    "tenant_email": {
        "tenant_email",
        "email",
    },
    "tenant_phone": {
        "tenant_phone",
        "phone",
    },
    "seats_used": {
        "seats_used",
        "seats",
    },
}


REQUIRED_ANY_OF = (
    "company_name",
    "tenant_name",
)


# ---------------------------------------------------------------------------
# Validated row
# ---------------------------------------------------------------------------

class CompanyRow(BaseModel):
    """One validated, normalized input row."""

    model_config = ConfigDict(extra="forbid")

    row_number: int

    company_name: str
    normalized_name: str

    website: str | None = None
    domain: str | None = None
    linkedin_url: str | None = None

    industry: str | None = None
    employee_count: int | None = None

    city: str | None = None
    state: str | None = None
    area: str | None = None
    address: str | None = None

    latitude: float | None = None
    longitude: float | None = None

    coworking_operator: str | None = None
    branch_name: str | None = None
    building_name: str | None = None

    seats_used: int | None = None
    tenant_email: str | None = None
    tenant_phone: str | None = None

    @field_validator("latitude")
    @classmethod
    def _lat(
        cls,
        v: float | None,
    ) -> float | None:
        if v is not None and not -90 <= v <= 90:
            raise ValueError("latitude out of range")
        return v

    @field_validator("longitude")
    @classmethod
    def _lng(
        cls,
        v: float | None,
    ) -> float | None:
        if v is not None and not -180 <= v <= 180:
            raise ValueError("longitude out of range")
        return v

    @model_validator(mode="after")
    def _coords_paired(self) -> "CompanyRow":
        if (
            self.latitude is None
        ) != (
            self.longitude is None
        ):
            raise ValueError(
                "latitude and longitude must be provided together"
            )

        return self

    @classmethod
    def from_raw(
        cls,
        row_number: int,
        raw: dict[str, object],
        default_region: str = "IN",
    ) -> "CompanyRow":
        """
        Normalize a canonical-keyed raw dict into a validated row.

        The coworking operator is taken from an explicit source column when
        available. Otherwise it is automatically detected from company_name.
        """

        name = (
            n.clean_text(raw.get("company_name"))
            or n.clean_text(raw.get("tenant_name"))
        )

        if not name:
            raise ValueError(
                "missing company_name/tenant_name"
            )

        website = n.normalize_url(
            raw.get("website")
            or raw.get("tenant_website")
        )

        # Explicit operator from source takes priority.
        explicit_operator = n.clean_text(
            raw.get("coworking_operator")
        )

        # Otherwise detect known operator from listing/company name.
        coworking_operator = (
            explicit_operator
            or n.detect_coworking_operator(name)
        )

        return cls(
            row_number=row_number,

            company_name=name,

            normalized_name=n.normalize_company_name(
                name
            ),

            website=website,

            domain=n.extract_domain(
                website
            ),

            linkedin_url=n.normalize_linkedin_url(
                raw.get("linkedin_url")
            ),

            industry=n.clean_text(
                raw.get("industry")
            ),

            employee_count=n.parse_int(
                raw.get("employee_count")
            ),

            city=n.normalize_city(
                raw.get("city")
            ),

            state=n.clean_text(
                raw.get("state")
            ),

            area=n.clean_text(
                raw.get("area")
            ),

            address=n.clean_text(
                raw.get("address")
            ),

            latitude=n.parse_float(
                raw.get("latitude")
            ),

            longitude=n.parse_float(
                raw.get("longitude")
            ),

            coworking_operator=coworking_operator,

            branch_name=n.clean_text(
                raw.get("branch_name")
            ),

            building_name=n.clean_text(
                raw.get("building_name")
            ),

            seats_used=n.parse_int(
                raw.get("seats_used")
            ),

            tenant_email=n.normalize_email(
                raw.get("tenant_email")
            ),

            tenant_phone=n.normalize_phone(
                raw.get("tenant_phone"),
                default_region,
            ),
        )