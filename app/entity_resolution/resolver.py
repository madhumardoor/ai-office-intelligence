"""Location-aware company/entity resolution for coworking data.

Rules:
- Same operator + same normalized address -> existing company/location.
- Same operator + different physical address -> same canonical company,
  and the ingestion pipeline creates a new CompanyLocation.
- Same coordinates can be used as a fallback when an address is missing.
- Different operators are never merged merely because they share an address.
- Exact domain / LinkedIn remain the strongest company identifiers.
- Conflicting strong identifiers never auto-match.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from enum import StrEnum

from sqlalchemy import or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.entity_resolution.similarity import (
    Candidate,
    FeatureWeights,
    MatchScore,
    name_similarity,
    score_pair,
)
from app.models import Company, CompanyAlias, CompanyIdentifier, CompanyLocation

MAX_BLOCK_CANDIDATES = 25
TRIGRAM_MIN = 0.35
COORD_TOLERANCE = 0.0015
MAX_OPERATOR_BLOCK_CANDIDATES = 250

_GENERIC_NAME_TOKENS = {
    "a", "an", "the", "co", "company", "companies", "cowork", "coworks",
    "coworking", "co-working", "workspace", "workspaces", "work", "working",
    "space", "spaces", "office", "offices", "shared", "virtual", "center",
    "centre", "business", "hub", "solutions", "services", "facility",
    "facilities", "management", "india", "private", "limited", "pvt", "ltd",
    "llp", "inc", "incorporated", "group",
}

_WEAK_SINGLE_TOKENS = {"next", "one", "hq", "ri", "pro", "my", "go"}


class Decision(StrEnum):
    AUTO_MATCH = "auto_match"
    NEEDS_REVIEW = "needs_review"
    NEW = "new"


@dataclass(slots=True)
class Resolution:
    decision: Decision
    company_id: uuid.UUID | None = None
    score: MatchScore | None = None
    review_candidates: list[tuple[uuid.UUID, MatchScore]] = field(default_factory=list)


def _normalize_address(value: str | None) -> str:
    """Normalize address for exact physical-location comparison."""
    if not value:
        return ""
    return "".join(ch for ch in value.casefold() if ch.isalnum())


def _name_tokens(name: str | None) -> set[str]:
    """Extract meaningful name tokens and ignore generic location/type words."""
    if not name:
        return set()

    result: set[str] = set()
    for token in re.findall(r"[a-z0-9]+", name.casefold()):
        if token in _GENERIC_NAME_TOKENS:
            continue
        if token.isdigit() or len(token) < 3:
            continue
        result.add(token)
    return result


def _same_operator_name(a: str | None, b: str | None) -> bool:
    """Identify operator-name continuity while allowing branch suffixes."""
    if not a or not b:
        return False

    na = re.sub(r"\s+", " ", a.casefold()).strip()
    nb = re.sub(r"\s+", " ", b.casefold()).strip()

    if na == nb:
        return True

    ta = _name_tokens(a)
    tb = _name_tokens(b)
    if not ta or not tb:
        return False

    # Examples: WeWork <-> WeWork Manyata, UrbanVault <-> Urban Vault HSR Layout.
    if ta.issubset(tb) or tb.issubset(ta):
        smaller = ta if len(ta) <= len(tb) else tb
        if len(smaller) >= 2:
            return True
        token = next(iter(smaller))
        if len(token) >= 6 and token not in _WEAK_SINGLE_TOKENS:
            return True

    # Conservative fuzzy fallback.
    clean_a = re.sub(r"[^a-z0-9 ]", " ", na)
    clean_b = re.sub(r"[^a-z0-9 ]", " ", nb)
    return name_similarity(clean_a, clean_b) >= 0.94


def _same_coordinates(
    a_lat: float | None,
    a_lng: float | None,
    b_lat: float | None,
    b_lng: float | None,
) -> bool:
    if None in (a_lat, a_lng, b_lat, b_lng):
        return False
    return (
        abs(float(a_lat) - float(b_lat)) <= COORD_TOLERANCE
        and abs(float(a_lng) - float(b_lng)) <= COORD_TOLERANCE
    )


def _location_match_type(incoming: Candidate, location: CompanyLocation) -> str:
    """Return address/coordinates when the incoming location matches."""
    incoming_address = _normalize_address(incoming.address)
    existing_address = _normalize_address(location.address)

    if incoming_address and existing_address and incoming_address == existing_address:
        return "address"

    if _same_coordinates(
        incoming.latitude,
        incoming.longitude,
        location.latitude,
        location.longitude,
    ):
        return "coordinates"

    return ""


class EntityResolver:
    def __init__(
        self,
        auto_threshold: float = 0.92,
        review_threshold: float = 0.75,
        weights: FeatureWeights | None = None,
    ) -> None:
        if review_threshold >= auto_threshold:
            raise ValueError("review_threshold must be below auto_threshold")
        self.auto = auto_threshold
        self.review = review_threshold
        self.weights = weights or FeatureWeights()

    async def _block(self, session: AsyncSession, inc: Candidate) -> list[Company]:
        """Recall-oriented candidate generation."""
        conds = []

        if inc.domain:
            conds.append(Company.domain == inc.domain)
            ident = select(CompanyIdentifier.company_id).where(
                CompanyIdentifier.id_type == "domain",
                CompanyIdentifier.id_value == inc.domain,
            )
            conds.append(Company.id.in_(ident))

        if inc.linkedin_url:
            conds.append(Company.linkedin_url == inc.linkedin_url)
            ident = select(CompanyIdentifier.company_id).where(
                CompanyIdentifier.id_type == "linkedin",
                CompanyIdentifier.id_value == inc.linkedin_url,
            )
            conds.append(Company.id.in_(ident))

        if inc.normalized_name:
            conds.append(Company.normalized_name == inc.normalized_name)
            alias_hit = select(CompanyAlias.company_id).where(
                CompanyAlias.normalized_alias == inc.normalized_name,
            )
            conds.append(Company.id.in_(alias_hit))
            conds.append(
                text(
                    "similarity(companies.normalized_name, :nm) >= :floor"
                ).bindparams(nm=inc.normalized_name, floor=TRIGRAM_MIN)
            )

        # Coworking rows frequently use branch/location names such as
        # "WeWork Galaxy", "WeWork Bellandur", or "Awfis Krish Cubical".
        # The operator is the stable company-level identity, so include
        # operator-prefixed company names in candidate generation. This is
        # only a blocking/retrieval rule; the final decision still requires
        # the operator/location logic below.
        if inc.operator_name:
            operator_key = re.sub(
                r"[^a-z0-9]+",
                " ",
                inc.operator_name.casefold(),
            ).strip()

            if operator_key:
                conds.append(
                    Company.normalized_name.ilike(f"{operator_key}%")
                )

        if not conds:
            return []

        candidate_limit = (
            MAX_OPERATOR_BLOCK_CANDIDATES
            if inc.operator_name
            else MAX_BLOCK_CANDIDATES
        )

        stmt = (
            select(Company)
            .where(
                Company.deleted_at.is_(None),
                Company.merged_into_id.is_(None),
                or_(*conds),
            )
            .limit(candidate_limit)
        )
        return list((await session.execute(stmt)).scalars())

    async def _get_locations(
        self,
        session: AsyncSession,
        companies: list[Company],
    ) -> dict[uuid.UUID, list[CompanyLocation]]:
        if not companies:
            return {}

        result = await session.execute(
            select(CompanyLocation).where(
                CompanyLocation.company_id.in_([c.id for c in companies])
            )
        )

        grouped: dict[uuid.UUID, list[CompanyLocation]] = {}
        for location in result.scalars().all():
            grouped.setdefault(location.company_id, []).append(location)
        return grouped

    @staticmethod
    def _company_candidate(company: Company) -> Candidate:
        return Candidate(
            normalized_name=company.normalized_name,
            domain=company.domain,
            linkedin_url=company.linkedin_url,
            city=company.city,
        )

    @staticmethod
    def _location_candidate(company: Company, location: CompanyLocation) -> Candidate:
        return Candidate(
            normalized_name=company.normalized_name,
            domain=company.domain,
            linkedin_url=company.linkedin_url,
            city=location.city or company.city,
            address=location.address,
            latitude=location.latitude,
            longitude=location.longitude,
        )

    async def _score_company(
        self,
        incoming: Candidate,
        company: Company,
        locations: list[CompanyLocation],
    ) -> MatchScore:
        """Score against all known locations and retain the strongest one."""
        if not locations:
            return score_pair(
                incoming,
                self._company_candidate(company),
                self.weights,
            )

        best: MatchScore | None = None
        for location in locations:
            current = score_pair(
                incoming,
                self._location_candidate(company, location),
                self.weights,
            )
            if best is None or current.confidence > best.confidence:
                best = current

        assert best is not None
        return best

    async def resolve(self, session: AsyncSession, inc: Candidate) -> Resolution:
        companies = await self._block(session, inc)
        if not companies:
            return Resolution(Decision.NEW)

        locations_by_company = await self._get_locations(session, companies)
        candidates: list[tuple[Company, MatchScore, bool, str, bool]] = []

        has_physical_location = bool(
            inc.address
            or (inc.latitude is not None and inc.longitude is not None)
        )

        for company in companies:
            score = await self._score_company(
                inc,
                company,
                locations_by_company.get(company.id, []),
            )
            if score.hard_block:
                continue

            # Only apply coworking operator/branch rules when the incoming row
            # explicitly identifies an operator. This prevents ordinary tenant
            # companies such as "Zephyr Systems" from being treated as an
            # operator just because their company name is similar.
            same_operator = bool(
                inc.operator_name
                and _same_operator_name(
                    inc.operator_name,
                    company.normalized_name,
                )
            )

            location_match = ""
            for location in locations_by_company.get(company.id, []):
                matched = _location_match_type(inc, location)
                if matched == "address":
                    location_match = "address"
                    break
                if matched == "coordinates" and not location_match:
                    location_match = "coordinates"

            candidates.append(
                (
                    company,
                    score,
                    same_operator,
                    location_match,
                    has_physical_location,
                )
            )

        if not candidates:
            return Resolution(Decision.NEW)

        # Physical/company-aware priority comes before fuzzy confidence.
        candidates.sort(
            key=lambda item: (
                1 if item[2] and item[3] == "address" else 0,
                1 if item[2] and item[3] == "coordinates" else 0,
                1 if item[2] and item[4] else 0,
                item[1].confidence,
            ),
            reverse=True,
        )

        best_company, best_score, same_operator, location_match, _ = candidates[0]

        # Same operator + same address/coordinates = existing location.
        if same_operator and location_match in {"address", "coordinates"}:
            best_score.features["same_operator"] = True
            best_score.features["location_decision"] = "duplicate"
            best_score.confidence = max(best_score.confidence, 0.99)
            return Resolution(Decision.AUTO_MATCH, best_company.id, best_score)

        # Same operator + different address = SAME COMPANY, NEW BRANCH.
        # The pipeline's _upsert_location() creates the new CompanyLocation.
        if same_operator and has_physical_location:
            best_score.features["same_operator"] = True
            best_score.features["same_address"] = False
            best_score.features["location_decision"] = "new_branch"
            return Resolution(Decision.AUTO_MATCH, best_company.id, best_score)

        # Standard fuzzy resolution.
        candidates.sort(key=lambda item: item[1].confidence, reverse=True)
        best_company, best_score, _, _, _ = candidates[0]

        ambiguous = (
            len(candidates) > 1
            and (best_score.confidence - candidates[1][1].confidence) < 0.03
            and candidates[1][1].confidence >= self.review
        )

        if best_score.confidence >= self.auto and not ambiguous:
            return Resolution(Decision.AUTO_MATCH, best_company.id, best_score)

        if best_score.confidence >= self.review:
            reviews = [
                (company.id, score)
                for company, score, _, _, _ in candidates
                if score.confidence >= self.review
            ][:3]
            return Resolution(
                Decision.NEEDS_REVIEW,
                best_company.id,
                best_score,
                reviews,
            )

        return Resolution(Decision.NEW, None, best_score)
