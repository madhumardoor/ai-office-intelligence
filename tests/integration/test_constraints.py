import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

pytestmark = pytest.mark.integration


async def _company(session, name="Acme", domain="acme.example") -> str:
    r = await session.execute(text(
        "INSERT INTO companies (name, normalized_name, domain) VALUES (:n, :nn, :d) RETURNING id"),
        {"n": name, "nn": name.lower(), "d": domain})
    return str(r.scalar_one())


async def test_duplicate_live_domain_rejected(db_session) -> None:
    await _company(db_session, "A", "dup.example")
    with pytest.raises(IntegrityError):
        await _company(db_session, "B", "dup.example")


async def test_soft_deleted_company_frees_domain(db_session) -> None:
    cid = await _company(db_session, "A", "free.example")
    await db_session.execute(text("UPDATE companies SET deleted_at = now() WHERE id = :i"), {"i": cid})
    await _company(db_session, "B", "free.example")  # allowed: partial unique index


async def test_negative_employee_count_rejected(db_session) -> None:
    with pytest.raises(IntegrityError):
        await db_session.execute(text(
            "INSERT INTO companies (name, normalized_name, employee_count) VALUES ('X','x',-5)"))


async def test_signal_requires_valid_kind(db_session) -> None:
    cid = await _company(db_session, "S", "s.example")
    await db_session.execute(text(
        "INSERT INTO signal_definitions (code,label,weight,rationale) VALUES ('t','t',0.5,'r')"))
    with pytest.raises(IntegrityError):
        await db_session.execute(text(
            "INSERT INTO signals (company_id,signal_code,kind,evidence_text,evidence_hash,detected_on,confidence) "
            "VALUES (:c,'t','opinion','e','h',now(),'low')"), {"c": cid})