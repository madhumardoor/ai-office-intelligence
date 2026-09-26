import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.integration


async def test_ready_with_real_postgres_and_extensions(integration_client: AsyncClient) -> None:
    r = await integration_client.get("/health/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready"
    names = {c["name"]: c for c in body["components"]}
    assert names["postgres"]["ok"] is True
    assert names["extensions"]["ok"] is True
    for ext in ("postgis", "vector", "pg_trgm", "pgcrypto"):
        assert ext in names["extensions"]["detail"]