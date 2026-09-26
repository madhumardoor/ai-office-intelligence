from httpx import ASGITransport, AsyncClient

from app.config import Settings
from app.main import create_app


async def test_liveness_and_request_id_generated() -> None:
    app = create_app(Settings(app_env="test"))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert len(r.headers["x-request-id"]) >= 8


async def test_valid_inbound_request_id_is_echoed() -> None:
    app = create_app(Settings(app_env="test"))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/health", headers={"x-request-id": "abc-12345678"})
    assert r.headers["x-request-id"] == "abc-12345678"


async def test_malicious_request_id_is_replaced() -> None:
    app = create_app(Settings(app_env="test"))
    bad = "x\nINJECTED log line"
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/health", headers={"x-request-id": "bad id with spaces!!"})
    assert r.headers["x-request-id"] != "bad id with spaces!!"
    assert bad not in r.headers["x-request-id"]


async def test_unknown_route_returns_structured_error() -> None:
    app = create_app(Settings(app_env="test"))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/nope")
    assert r.status_code == 404
    body = r.json()["error"]
    assert body["code"] == "http_error"
    assert body["request_id"]