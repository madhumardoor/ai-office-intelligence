import json
import logging

from app.config.logging import JsonFormatter, request_id_ctx


def _record(msg: str, **extra: object) -> logging.LogRecord:
    rec = logging.LogRecord("t", logging.INFO, __file__, 1, msg, (), None)
    for k, v in extra.items():
        setattr(rec, k, v)
    return rec


def test_json_output_has_core_fields() -> None:
    out = json.loads(JsonFormatter().format(_record("hello")))
    assert out["message"] == "hello"
    assert out["level"] == "INFO"
    assert "timestamp" in out


def test_request_id_is_included() -> None:
    token = request_id_ctx.set("req-123456789")
    try:
        out = json.loads(JsonFormatter().format(_record("x")))
    finally:
        request_id_ctx.reset(token)
    assert out["request_id"] == "req-123456789"


def test_secret_keys_redacted() -> None:
    out = json.loads(
        JsonFormatter().format(_record("x", api_key="sk-abc", nested={"password": "p", "ok": 1}))
    )
    assert out["api_key"] == "***REDACTED***"
    assert out["nested"]["password"] == "***REDACTED***"
    assert out["nested"]["ok"] == 1