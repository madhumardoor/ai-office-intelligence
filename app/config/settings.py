"""Application settings loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal
from urllib.parse import quote_plus

from pydantic import Field, SecretStr, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed, validated application configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- App ---
    app_env: Literal["local", "test", "production"] = "local"
    app_name: str = "ai-office-intelligence"
    log_level: str = "INFO"
    log_json: bool = True

    # --- Database ---
    postgres_user: str = "postgres"
    postgres_password: SecretStr = Field(default=SecretStr(""))
    postgres_db: str = "ai-office-intelligence"
    postgres_host: str = "127.0.0.1"
    postgres_port: int = 5432
    agent_ro_password: SecretStr = Field(default=SecretStr(""))

    # --- Redis persistent memory ---
    redis_url: str = "redis://localhost:6379/0"

    redis_memory_ttl_seconds: int = Field(
        default=2592000,
        ge=60,
    )

    redis_memory_max_messages: int = Field(
        default=200,
        ge=10,
        le=5000,
    )

    redis_memory_max_turns: int = Field(
        default=100,
        ge=10,
        le=1000,
    )

    db_pool_size: int = Field(default=10, ge=1, le=100)
    db_max_overflow: int = Field(default=10, ge=0, le=100)
    db_pool_timeout_seconds: int = Field(default=30, ge=1)
    db_statement_timeout_ms: int = Field(default=15000, ge=100)

    # --- Ingestion / entity resolution ---
    ingest_checkpoint_every: int = Field(default=50, ge=1)
    er_auto_match_threshold: float = Field(default=0.92, ge=0, le=1)
    er_review_threshold: float = Field(default=0.75, ge=0, le=1)
    default_country_code: str = "IN"

    # --- Geocoding (optional) ---
    geocoding_enabled: bool = False
    geocoding_user_agent: str = "ai-office-intelligence/0.1"
    geocoding_min_interval_seconds: float = Field(
        default=1.1,
        ge=1.0,
    )

    # --- Embeddings ---
    embedding_provider: Literal[
        "local",
        "openai",
        "gemini",
    ] = "local"

    embedding_model: str = "BAAI/bge-small-en-v1.5"

    embedding_dimension: int = 384

    embedding_batch_size: int = Field(
        default=32,
        ge=1,
        le=256,
    )

    embedding_api_key: SecretStr = Field(
        default=SecretStr("")
    )

    # --- Retrieval / chunking ---
    reranker: Literal[
        "none",
        "cross_encoder",
    ] = "none"

    reranker_model: str = "BAAI/bge-reranker-base"

    retrieval_candidates_per_method: int = Field(
        default=30,
        ge=1,
        le=200,
    )

    retrieval_final_k: int = Field(
        default=8,
        ge=1,
        le=50,
    )

    chunk_target_tokens: int = Field(
        default=450,
        ge=100,
        le=2000,
    )

    chunk_overlap_tokens: int = Field(
        default=60,
        ge=0,
        le=500,
    )

    context_token_budget: int = Field(
        default=3000,
        ge=500,
        le=20000,
    )

    # --- LLM ---
    llm_provider: Literal[
        "gemini",
        "openai",
        "groq",
    ] = "groq"

    llm_api_key: SecretStr = Field(
        default=SecretStr("")
    )

    llm_model_fast: str = ""

    llm_model_strong: str = ""

    llm_timeout_seconds: float = Field(
        default=45.0,
        ge=1,
    )

    llm_max_retries: int = Field(
        default=3,
        ge=0,
        le=8,
    )

    llm_max_output_tokens: int = Field(
        default=1500,
        ge=64,
        le=16000,
    )

    llm_cache_enabled: bool = True

    llm_cache_max_entries: int = Field(
        default=500,
        ge=1,
    )

    llm_log_to_db: bool = True

    llm_pricing_json: str = ""

    @computed_field(repr=False)
    @property
    def database_url(self) -> str:
        """Async SQLAlchemy URL for the application database."""
        pw = quote_plus(
            self.postgres_password.get_secret_value()
        )

        return (
            f"postgresql+asyncpg://"
            f"{quote_plus(self.postgres_user)}:{pw}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @computed_field(repr=False)
    @property
    def agent_ro_database_url(self) -> str:
        """Async URL for the read-only agent role."""
        pw = quote_plus(
            self.agent_ro_password.get_secret_value()
        )

        return (
            f"postgresql+asyncpg://agent_ro:{pw}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    def model_dump(self, *args, **kwargs):
        """Prevent computed database URLs from exposing credentials."""
        exclude = kwargs.setdefault("exclude", set())

        if isinstance(exclude, set):
            exclude.update(
                {
                    "database_url",
                    "agent_ro_database_url",
                }
            )
        elif isinstance(exclude, dict):
            exclude["database_url"] = True
            exclude["agent_ro_database_url"] = True

        return super().model_dump(*args, **kwargs)


@lru_cache
def get_settings() -> Settings:
    """Return the cached settings singleton."""
    return Settings()