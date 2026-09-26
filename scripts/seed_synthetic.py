"""Generate a clearly-labelled SYNTHETIC Bangalore dataset for development and demos.

Everything here is fictional (names, tenants, hiring counts). Coordinates are approximate real
Bangalore localities so spatial queries behave realistically. Deterministic via --seed.

Usage:  uv run python -m scripts.seed_synthetic --companies 120 --seed 42
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import random
from datetime import UTC, date, datetime, timedelta

from geoalchemy2.elements import WKTElement
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import create_engine, create_session_factory
from app.models import (
    Company, CompanyAlias, CompanyEmployeeSnapshot, CompanyFunding, CompanyHiringSnapshot,
    CompanyLocation, CompanyNews, CoworkingBranch, CoworkingBuilding, CoworkingOperator,
    CoworkingTenant, Property, SignalDefinition, Source,
)
from app.models.enums import SourceType

# (area, lat, lng): approximate locality centres.
AREAS = [
    ("Whitefield", 12.9698, 77.7500), ("Koramangala", 12.9352, 77.6245),
    ("Indiranagar", 12.9784, 77.6408), ("HSR Layout", 12.9116, 77.6474),
    ("Electronic City", 12.8452, 77.6602), ("Manyata Tech Park", 13.0451, 77.6200),
    ("Bellandur", 12.9256, 77.6762), ("MG Road", 12.9756, 77.6069),
    ("Marathahalli", 12.9591, 77.6974), ("Hebbal", 13.0358, 77.5970),
]
OPERATORS = ["Hive Spaces", "OrbitWork", "Nimbus Cowork", "BlueDesk"]
INDUSTRIES = ["SaaS", "Fintech", "Healthtech", "E-commerce", "AI/ML", "Logistics", "EdTech", "Cybersecurity"]
NAME_A = ["Nova", "Quanta", "Zephyr", "Lumen", "Vertex", "Aether", "Cobalt", "Ember", "Pixel", "Stratus", "Helix", "Onyx"]
NAME_B = ["Labs", "Systems", "Technologies", "Analytics", "Networks", "Robotics", "Cloud", "Dynamics"]

# Default signal weights. Each has a rationale: weights are hypotheses, not validated truths.
SIGNAL_DEFS = [
    ("hiring_acceleration", "Hiring acceleration", 0.30, 45,
     "Rising open roles in a city imply near-term headcount growth, the most direct precursor to space needs."),
    ("headcount_growth", "Headcount growth", 0.25, 120,
     "Measured increase between employee snapshots shows realised growth, not just intent."),
    ("recent_funding", "Recent funding", 0.20, 180,
     "Fresh capital commonly funds hiring/expansion, but timing varies widely; hence lower certainty."),
    ("expansion_announcement", "Expansion announcement", 0.25, 120,
     "An explicit statement of new office/city expansion is strong intent evidence."),
    ("multi_location_presence", "Multi-location presence", 0.10, 365,
     "Occupying several coworking sites may indicate outgrowing a single space."),
    ("leadership_hiring", "Leadership hiring", 0.10, 90,
     "Senior hires often precede team build-outs."),
]


def _hash(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def _jitter(rng: random.Random, lat: float, lng: float, km: float = 1.5) -> tuple[float, float]:
    # ~0.009 degrees latitude per km; longitude scaled for ~13N.
    return lat + rng.uniform(-km, km) * 0.009, lng + rng.uniform(-km, km) * 0.0092


def _point(lat: float, lng: float) -> WKTElement:
    return WKTElement(f"POINT({lng} {lat})", srid=4326)  # WKT is lon lat


async def seed(session: AsyncSession, n_companies: int, seed_value: int) -> dict[str, int]:
    rng = random.Random(seed_value)
    today = date.today()
    now = datetime.now(UTC)

    src = (await session.execute(select(Source).where(
        Source.source_type == SourceType.SYNTHETIC, Source.name == "synthetic_seed"))).scalar_one_or_none()
    if src is None:
        src = Source(source_type=SourceType.SYNTHETIC, name="synthetic_seed", reliability=0.0,
                     extra={"warning": "FICTIONAL DATA. Not for real business decisions."})
        session.add(src)
        await session.flush()

    existing = {d.code for d in (await session.execute(select(SignalDefinition))).scalars()}
    for code, label, weight, half_life, rationale in SIGNAL_DEFS:
        if code not in existing:
            session.add(SignalDefinition(code=code, label=label, weight=weight,
                                         half_life_days=half_life, rationale=rationale))

    operators = {}
    for name in OPERATORS:
        op = (await session.execute(select(CoworkingOperator).where(CoworkingOperator.name == name))).scalar_one_or_none()
        if op is None:
            op = CoworkingOperator(name=name, source_id=src.id, retrieved_at=now)
            session.add(op)
            await session.flush()
        operators[name] = op

    branches: list[CoworkingBranch] = []
    for area, lat, lng in AREAS:
        blat, blng = _jitter(rng, lat, lng, 0.8)
        bname = f"{area} Business Centre (synthetic)"
        b = (await session.execute(select(CoworkingBuilding).where(
            CoworkingBuilding.name == bname, CoworkingBuilding.city == "Bengaluru"))).scalar_one_or_none()
        if b is None:
            b = CoworkingBuilding(name=bname, area=area, city="Bengaluru", latitude=blat, longitude=blng,
                                  geog=_point(blat, blng), total_floors=rng.randint(6, 20),
                                  source_id=src.id, retrieved_at=now)
            session.add(b)
            await session.flush()
        for op_name in rng.sample(OPERATORS, k=rng.randint(1, 2)):
            br = (await session.execute(select(CoworkingBranch).where(
                CoworkingBranch.operator_id == operators[op_name].id, CoworkingBranch.building_id == b.id,
                CoworkingBranch.name == f"{op_name} {area}"))).scalar_one_or_none()
            if br is None:
                br = CoworkingBranch(operator_id=operators[op_name].id, building_id=b.id,
                                     name=f"{op_name} {area}", seat_capacity=rng.choice([150, 300, 500]),
                                     source_id=src.id, retrieved_at=now)
                session.add(br)
                await session.flush()
            branches.append(br)

        p_lat, p_lng = _jitter(rng, lat, lng, 1.0)
        pname = f"{area} Tech Park (synthetic)"
        if not (await session.execute(select(Property).where(Property.name == pname))).scalar_one_or_none():
            session.add(Property(name=pname, property_type="tech_park", area=area, city="Bengaluru",
                                 latitude=p_lat, longitude=p_lng, geog=_point(p_lat, p_lng),
                                 total_area_sqft=rng.randint(200_000, 1_500_000),
                                 source_id=src.id, retrieved_at=now))

    counts = {"companies": 0, "tenancies": 0, "news": 0}
    used_names: set[str] = set()
    for i in range(n_companies):
        for _ in range(20):
            base = f"{rng.choice(NAME_A)} {rng.choice(NAME_B)}"
            if base not in used_names:
                used_names.add(base)
                break
        else:
            base = f"{base} {i}"
        slug = base.lower().replace(" ", "")
        domain = f"{slug}.example"  # .example is reserved (RFC 2606): guaranteed non-real
        area, lat, lng = rng.choice(AREAS)
        if (await session.execute(select(Company).where(Company.domain == domain))).scalar_one_or_none():
            continue
        growth_profile = rng.choice(["fast", "steady", "flat"])
        base_emp = rng.randint(15, 400)
        company = Company(
            name=f"{base} Pvt Ltd", normalized_name=base.lower(), website=f"https://{domain}", domain=domain,
            industry=rng.choice(INDUSTRIES), city="Bengaluru", state="Karnataka",
            employee_count=base_emp, description=f"{base} is a fictional {rng.choice(INDUSTRIES)} company (synthetic).",
            source_id=src.id, retrieved_at=now, raw_payload={"synthetic": True, "profile": growth_profile},
        )
        session.add(company)
        await session.flush()
        counts["companies"] += 1

        session.add(CompanyAlias(company_id=company.id, alias=base, normalized_alias=base.lower(), source_id=src.id))
        clat, clng = _jitter(rng, lat, lng)
        session.add(CompanyLocation(company_id=company.id, label="Bangalore office", area=area, city="Bengaluru",
                                    latitude=clat, longitude=clng, geog=_point(clat, clng), is_primary=True,
                                    source_id=src.id, retrieved_at=now))

        # Headcount history: 3 snapshots showing the growth profile.
        factor = {"fast": 1.35, "steady": 1.08, "flat": 1.0}[growth_profile]
        for months_ago, mult in ((6, 1 / factor**2), (3, 1 / factor), (0, 1.0)):
            session.add(CompanyEmployeeSnapshot(
                company_id=company.id, observed_on=today - timedelta(days=30 * months_ago),
                employee_count=max(1, int(base_emp * mult)), location_scope="Bangalore",
                source_id=src.id, retrieved_at=now))

        open_roles = {"fast": rng.randint(15, 40), "steady": rng.randint(3, 12), "flat": rng.randint(0, 3)}[growth_profile]
        session.add(CompanyHiringSnapshot(
            company_id=company.id, observed_on=today, city="Bangalore", open_roles=open_roles,
            leadership_roles=rng.randint(0, 3) if growth_profile == "fast" else 0,
            evidence_url=f"https://{domain}/careers", source_id=src.id, retrieved_at=now))

        if growth_profile == "fast" and rng.random() < 0.6:
            session.add(CompanyFunding(
                company_id=company.id, announced_on=today - timedelta(days=rng.randint(10, 150)),
                round_name=rng.choice(["Seed", "Series A", "Series B"]), amount_usd=rng.choice([2, 5, 12, 25]) * 1_000_000,
                evidence_url=f"https://{domain}/news/funding", source_id=src.id, retrieved_at=now))
        if growth_profile == "fast" and rng.random() < 0.5:
            url = f"https://{domain}/news/expansion"
            session.add(CompanyNews(
                company_id=company.id, title=f"{base} announces expansion of Bangalore team (synthetic)",
                url=url, url_hash=_hash(url), published_at=now - timedelta(days=rng.randint(1, 60)),
                snippet="Synthetic news item for development only.", category="expansion",
                source_id=src.id, retrieved_at=now))
            counts["news"] += 1

        # ~60% are coworking tenants; fast growers sometimes hold 2 branches.
        if rng.random() < 0.6 and branches:
            k = 2 if growth_profile == "fast" and rng.random() < 0.4 else 1
            for br in rng.sample(branches, k=min(k, len(branches))):
                first = today - timedelta(days=rng.randint(60, 700))
                session.add(CoworkingTenant(
                    company_id=company.id, branch_id=br.id, seats_used=max(2, base_emp // rng.randint(4, 10)),
                    first_seen_on=first, last_seen_on=today, source_id=src.id, retrieved_at=now,
                    contact_email=f"hello@{domain}", contact_phone="+91-00000-00000"))
                counts["tenancies"] += 1

    await session.commit()
    return counts


async def main(n: int, seed_value: int) -> None:
    engine = create_engine(get_settings())
    factory = create_session_factory(engine)
    async with factory() as session:
        result = await seed(session, n, seed_value)
    await engine.dispose()
    print(f"Seeded SYNTHETIC data: {result}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--companies", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    asyncio.run(main(args.companies, args.seed))