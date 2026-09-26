from pydantic import SecretStr

from app.config import Settings


def test_database_url_built_and_escaped() -> None:
    s = Settings(
        postgres_user="u",
        postgres_password=SecretStr("p@ss/word"),
        postgres_host="h",
        postgres_port=5433,
        postgres_db="d",
    )
    assert s.database_url == "postgresql+asyncpg://u:p%40ss%2Fword@h:5433/d"


def test_secrets_not_in_repr() -> None:
    s = Settings(postgres_password=SecretStr("supersecret"))
    assert "supersecret" not in repr(s)
    assert "supersecret" not in str(s.model_dump())


def test_agent_ro_url_uses_readonly_role() -> None:
    s = Settings(agent_ro_password=SecretStr("ro"))
    assert s.agent_ro_database_url.startswith("postgresql+asyncpg://agent_ro:")