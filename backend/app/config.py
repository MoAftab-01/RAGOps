"""Application configuration.

Every tunable in RAGOps is resolved here so that nothing else in the codebase
reads ``os.environ`` directly. Values come from environment variables (or a
local ``.env`` file), which is what keeps secrets out of version control.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _find_env_file() -> str | Path:
    """Locate the ``.env`` file regardless of the process working directory.

    The server, Alembic, and ``scripts/*.py`` all run from different
    directories (``backend/``, the repo root, ...), and a plain relative
    ``env_file=".env"`` silently finds nothing as soon as the CWD moves --
    which shows up as "the database password is wrong" even though the value is
    sitting in a file two levels up. Search upward from this file so every
    entry point resolves the same configuration.
    """
    here = Path(__file__).resolve()
    for candidate in (here.parents[2] / ".env", here.parents[1] / ".env", here.parent / ".env"):
        if candidate.is_file():
            return candidate
    # Nothing found: return a path that does not exist rather than raising, so
    # a clean checkout still boots from environment variables alone.
    return here.parents[2] / ".env"


class Settings(BaseSettings):
    """Runtime settings, populated from environment variables / ``.env``."""

    # `case_sensitive` is deliberately NOT enabled. In pydantic-settings it
    # switches matching from "field name" to "exact env-var spelling", which
    # means the field `database_url` would only ever be filled by an env var
    # literally named `database_url` -- every SCREAMING_SNAKE variable we
    # document in .env.example (DATABASE_URL, REDIS_URL, ...) would be
    # silently ignored. The default case-insensitive matching resolves both
    # `DATABASE_URL` and `database_url` to the same field, which is what we
    # want for an uppercase env-var convention.
    model_config = SettingsConfigDict(
        env_file=_find_env_file(),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------
    app_name: str = "RAGOps"
    environment: Literal["development", "test", "production"] = "development"
    debug: bool = True
    log_level: str = "INFO"
    api_v1_prefix: str = "/api"

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    # The host port here must match the port docker-compose.yml *publishes*,
    # which is `PG_PORT` (default 55432), not the container's internal 5432.
    # A mismatch here does not fail loudly -- Alembic reports a connection
    # refusal and the app reports `degraded` -- so it is stated twice rather
    # than left implicit. Override PG_PORT only if you also override these.
    database_url: str = "postgresql+asyncpg://ragops:ragops@localhost:55432/ragops"
    database_sync_url: str = (
        "postgresql+psycopg2://ragops:ragops@localhost:55432/ragops"
    )
    db_pool_size: int = 10
    db_max_overflow: int = 20
    db_echo: bool = False

    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 60

    # ------------------------------------------------------------------
    # Security
    # ------------------------------------------------------------------
    # A static key that keeps local setup frictionless. The collector routes
    # accept it via `X-API-Key`; analytics reads are open so the dashboard
    # works before anything is instrumented.
    ragops_api_key: str = "dev-key"
    auth_enabled: bool = True
    # Secret used to key the HMAC that stores per-company API keys. Empty by
    # default so a clean checkout boots with no extra configuration, and
    # `generate_api_key` warns when it is empty.
    #
    # What an empty pepper costs: the stored digests are computed from no
    # secret, so anyone holding a database dump can verify a *guessed* key
    # against them offline. They still cannot recover a key from a digest --
    # the key is 256 bits of CSPRNG output, and brute force is not a practical
    # attack against that. So the pepper defends a database backup from being
    # used as an offline oracle, not the key itself. Set it in production.
    #
    # The field name is load-bearing: `Settings` has no env_prefix and
    # case_sensitive is off, so this binds to API_KEY_PEPPER.
    api_key_pepper: str = ""
    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:8080",
            "http://localhost:3000",
        ]
    )

    # ------------------------------------------------------------------
    # LLM providers (local-first, no paid APIs required)
    # ------------------------------------------------------------------
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:3b"
    ollama_timeout_seconds: float = 120.0
    ollama_num_ctx: int = 8192
    ollama_temperature: float = 0.1

    # ------------------------------------------------------------------
    # Retrieval / embeddings
    # ------------------------------------------------------------------
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "auto"
    embedding_batch_size: int = 32

    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_enabled: bool = True
    reranker_top_n: int = 20

    vector_store_backend: Literal["faiss", "qdrant"] = "faiss"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "ragops_documents"

    # ------------------------------------------------------------------
    # Ingestion defaults for the demo RAG pipeline
    # ------------------------------------------------------------------
    chunk_size: int = 500
    chunk_overlap: int = 50
    default_top_k: int = 5
    hybrid_rrf_k: int = 60

    # ------------------------------------------------------------------
    # Telemetry ingestion
    # ------------------------------------------------------------------
    # Traces are buffered in Redis and flushed in batches, so instrumenting an
    # application does not add a synchronous write to its hot path.
    collector_buffer_size: int = 200
    collector_flush_interval_seconds: float = 2.0
    collector_enabled: bool = True

    # ------------------------------------------------------------------
    # Anomaly detection
    # ------------------------------------------------------------------
    anomaly_contamination: float = 0.02
    anomaly_min_samples: int = 50

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    # LLM-as-judge is opt-in. Deterministic/embedding metrics are the default
    # precisely so evaluation never depends on model behaviour.
    llm_judge_enabled: bool = False
    llm_judge_model: str = "qwen2.5:3b"
    judge_max_claims: int = 12
    grounding_similarity_threshold: float = 0.55

    # ------------------------------------------------------------------
    # Simulated pricing
    # ------------------------------------------------------------------
    # Local inference is free. RAGOps still models cost so that a team can
    # benchmark "what would this cost on a metered provider". Every surface
    # that shows a dollar figure is labelled as simulated.
    pricing_enabled: bool = True
    pricing_currency: str = "USD"

    # ------------------------------------------------------------------
    # Analytics caching
    # ------------------------------------------------------------------
    analytics_cache_ttl_seconds: int = 30
    analytics_cache_enabled: bool = True

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    knowledge_base_path: str = "evaluation/datasets/knowledge_base"
    evaluation_dataset_path: str = "evaluation/datasets/retrieval_eval.jsonl"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Cached so the environment is parsed once; tests call
    ``get_settings.cache_clear()`` after mutating the environment.
    """
    return Settings()


settings = get_settings()
