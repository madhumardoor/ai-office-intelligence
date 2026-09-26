import pytest
from sqlalchemy import text

from app.entity_resolution.resolver import Decision, EntityResolver
from app.entity_resolution.similarity import Candidate

pytestmark = pytest.mark.integration


async def _add(session, name: str, norm: str, domain: str | None, city: str = "Bengaluru") -> None:
    await session.execute(
        text("INSERT INTO companies (name, normalized_name, domain, city) VALUES (:n, :nn, :d, :c)"),
        {"n": name, "nn": norm, "d": domain, "c": city},
    )


async def test_same_domain_auto_matches(db_session) -> None:
    await _add(db_session, "Nova Labs Pvt Ltd", "nova labs", "nova.example")
    r = await EntityResolver().resolve(db_session, Candidate("nova", "nova.example", None, "Bengaluru"))
    assert r.decision == Decision.AUTO_MATCH


async def test_similar_name_different_domain_goes_to_review_not_merge(db_session) -> None:
    await _add(db_session, "Zephyr Systems", "zephyr", "zephyr.example")
    r = await EntityResolver().resolve(
        db_session, Candidate("zephyr", "zephyr-systems.example", None, "Bengaluru")
    )
    assert r.decision == Decision.NEEDS_REVIEW


async def test_unrelated_company_is_new(db_session) -> None:
    await _add(db_session, "Helix Cloud", "helix cloud", "helix.example")
    r = await EntityResolver().resolve(
        db_session, Candidate("quanta analytics", "quanta.example", None, "Bengaluru")
    )
    assert r.decision == Decision.NEW


async def test_conflicting_strong_identifiers_never_match(db_session) -> None:
    await db_session.execute(text(
        "INSERT INTO companies (name, normalized_name, domain, linkedin_url, city) "
        "VALUES ('Acme One','acme','acme-one.example',"
        "'https://www.linkedin.com/company/acme-one','Bengaluru')"
    ))
    r = await EntityResolver().resolve(
        db_session,
        Candidate("acme", "acme-two.example", "https://www.linkedin.com/company/acme-two", "Bengaluru"),
    )
    assert r.decision == Decision.NEW


async def test_soft_deleted_companies_are_ignored(db_session) -> None:
    await db_session.execute(text(
        "INSERT INTO companies (name, normalized_name, domain, deleted_at) "
        "VALUES ('Gone','gone','gone.example', now())"
    ))
    r = await EntityResolver().resolve(db_session, Candidate("gone", "gone.example", None, None))
    assert r.decision == Decision.NEW


def test_threshold_order_validated() -> None:
    with pytest.raises(ValueError):
        EntityResolver(auto_threshold=0.7, review_threshold=0.8)