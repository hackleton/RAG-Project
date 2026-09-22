"""Application settings.

All configuration comes from the environment (12-factor). Nothing in this file
may carry a production-usable default: `Settings.validate_production_safety`
refuses to boot a non-development environment that is still using dev values.

v1 runs entirely on the developer's machine - SQL Server, local file storage,
local extraction, local embedding and reranking models, a local LLM. Every one
of those sits behind a provider interface, so swapping in Azure Blob Storage,
Document Intelligence, AI Search and Azure OpenAI later is a configuration and
adapter change rather than a rewrite. See docs/decisions.md entry 11.
"""

from __future__ import annotations

import json
import secrets
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import AnyHttpUrl, BeforeValidator, Field, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# Sentinel used by .env.example and docker-compose for local development only.
DEV_JWT_SECRET = "dev-secret-change-me"


class AppEnv(StrEnum):
    LOCAL = "local"
    CI = "ci"
    STAGING = "staging"
    PRODUCTION = "production"


class LLMProvider(StrEnum):
    OLLAMA = "ollama"
    # Deterministic stub: composes an answer from the retrieved context without
    # a model. Keeps the pipeline end-to-end testable in CI, where no LLM runs.
    ECHO = "echo"


class EmbeddingProviderName(StrEnum):
    FASTEMBED = "fastembed"
    # Deterministic hashed embeddings. No model download, so the test suite and
    # CI exercise the real retrieval path without a 130MB dependency.
    HASH = "hash"


class PlannerName(StrEnum):
    """Which router decides the agent's tool calls."""

    # Deterministic intent rules. No model, so it is reproducible and runs in
    # CI - and it is the fallback whenever the LLM planner returns nonsense.
    RULE = "rule"
    # The model picks tools from the registry's catalogue.
    LLM = "llm"
    # LLM when a real model is configured, rules otherwise. The echo provider
    # cannot route, so "auto" avoids a config that silently does nothing.
    AUTO = "auto"


def _split_csv(value: object) -> object:
    """Accept both `a,b` and `["a","b"]` for list-valued env vars."""
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            return json.loads(text)
        return [item.strip() for item in text.split(",") if item.strip()]
    return value


# NoDecode stops pydantic-settings JSON-decoding the raw env value before
# validation runs; without it `CORS_ORIGINS=http://a,http://b` is a parse error.
CsvList = Annotated[list[str], NoDecode, BeforeValidator(_split_csv)]


def _empty_to_none(value: object) -> object:
    """Treat an unset-but-present env var as absent.

    `.env.example` lists optional keys with empty values so the file documents
    them; an empty string must mean "not configured", not "invalid value".
    """
    if isinstance(value, str) and not value.strip():
        return None
    return value


OptionalStr = Annotated[str | None, BeforeValidator(_empty_to_none)]
OptionalUrl = Annotated[AnyHttpUrl | None, BeforeValidator(_empty_to_none)]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Application -----------------------------------------------------
    app_env: AppEnv = AppEnv.LOCAL
    app_name: str = "Healthcare Intelligence Copilot API"
    api_prefix: str = "/api/v1"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["json", "console"] = "json"
    debug: bool = False

    # --- HTTP ------------------------------------------------------------
    cors_origins: CsvList = Field(default_factory=lambda: ["http://localhost:3000"])

    # --- Datastores ------------------------------------------------------
    # SQL Server over the async ODBC driver. Alembic uses the same URL.
    database_url: str = (
        "mssql+aioodbc://copilot_app:Local_Dev_App_Password_1@localhost:1433/copilot"
        "?driver=ODBC+Driver+18+for+SQL+Server&TrustServerCertificate=yes"
    )
    # SELECT-only login used by the agent's SQL tool. Never the app's own login.
    readonly_database_url: OptionalStr = None
    readonly_query_timeout_seconds: int = 10
    database_pool_size: int = 5
    database_max_overflow: int = 10
    database_echo: bool = False
    redis_url: str = "redis://localhost:6379/0"

    # --- Vector store (stands in for the Azure AI Search index) ----------
    qdrant_url: AnyHttpUrl = AnyHttpUrl("http://localhost:6333")
    qdrant_api_key: OptionalStr = None
    qdrant_collection: str = "healthcare-chunks"
    # Set to ":memory:" to run Qdrant embedded in-process. The test suite
    # uses it: named dense + sparse vectors, the IDF modifier and payload
    # filters all behave as they do against the server, so the suite covers
    # the real retrieval path without a container.
    qdrant_location: OptionalStr = None
    qdrant_timeout_seconds: float = 15.0

    # --- Local document storage (stands in for Azure Blob Storage) --------
    storage_dir: Path = Path("./.data/storage")
    max_upload_mb: int = 25

    # --- Local models (stand in for Azure OpenAI / AI Search reranker) -----
    embedding_provider: EmbeddingProviderName = EmbeddingProviderName.FASTEMBED
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimensions: int = 384
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    model_cache_dir: Path = Path("./.data/models")

    # --- Local LLM (stands in for an Azure OpenAI chat deployment) --------
    llm_provider: LLMProvider = LLMProvider.ECHO
    ollama_base_url: AnyHttpUrl = AnyHttpUrl("http://localhost:11434")
    llm_model: str = "llama3.1:8b"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024

    # --- Retrieval defaults ----------------------------------------------
    # Project plan section 14 treats every one of these as an experiment
    # variable, not a constant. They are configuration so a run can sweep them.
    chunk_target_tokens: int = 512
    chunk_overlap_tokens: int = 64
    retrieval_top_k: int = 5
    retrieval_candidate_pool: int = 50
    rrf_k: int = 60
    enable_reranker: bool = True
    # How many fused candidates the cross-encoder scores. It is the
    # dominant cost in a request, and linear in this number.
    rerank_top_n: int = 30
    # Retrieval strategy, so the section 14 experiment 'vector-only vs
    # hybrid' is a configuration change rather than a code change.
    retrieval_mode: Literal["hybrid", "dense", "sparse"] = "hybrid"
    # Discard candidates the reranker scores below this. None = keep all.
    rerank_score_threshold: float | None = None

    # --- Agent ------------------------------------------------------------
    agent_planner: PlannerName = PlannerName.AUTO
    # Ceiling on tool calls per run. The router is untrusted input, so the
    # bound is enforced by the dispatcher rather than requested in the prompt.
    agent_max_tool_calls: int = 4
    # Free-form NL-to-SQL. Off by default: it is the one tool whose input is a
    # generated program, and a deployment should opt into that deliberately.
    sql_tool_enabled: bool = False
    # Hard row cap applied by the executor, not by the generated SQL.
    # The table allow-list is deliberately *not* configuration: it is derived
    # per role in `app.tools.sql_guard`, because a table that can be added by
    # environment variable can be added without anyone deciding how to scope it.
    sql_max_rows: int = 200

    # --- Auth ------------------------------------------------------------
    jwt_secret_key: str = DEV_JWT_SECRET
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    # bcrypt cost. 12 is the production floor; tests override it to keep the
    # suite fast (bcrypt is intentionally slow).
    bcrypt_rounds: int = 12

    @property
    def is_local(self) -> bool:
        return self.app_env in (AppEnv.LOCAL, AppEnv.CI)

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def effective_planner(self) -> PlannerName:
        """Resolve `auto`. The echo provider cannot make routing decisions."""
        if self.agent_planner is not PlannerName.AUTO:
            return self.agent_planner
        return PlannerName.RULE if self.llm_provider is LLMProvider.ECHO else PlannerName.LLM

    @property
    def sql_session_url(self) -> str:
        """The URL the SQL tool connects with.

        Falls back to the application login only in local/CI, where there is
        often no second login to have - SQLite has no roles at all. The
        application-layer guard still runs, and `validate_production_safety`
        refuses the fallback anywhere else.
        """
        return self.readonly_database_url or self.database_url

    @model_validator(mode="after")
    def validate_production_safety(self) -> Settings:
        if self.chunk_overlap_tokens >= self.chunk_target_tokens:
            raise ValueError("CHUNK_OVERLAP_TOKENS must be smaller than CHUNK_TARGET_TOKENS.")
        if self.retrieval_top_k > self.retrieval_candidate_pool:
            raise ValueError("RETRIEVAL_TOP_K cannot exceed RETRIEVAL_CANDIDATE_POOL.")

        if self.is_local:
            return self

        # Outside local/CI the SQL tool must have a genuinely separate login.
        # The in-process guard rejects everything that is not a scoped SELECT,
        # but it is one layer; the database privilege is the one that still
        # holds if the guard has a bug.
        if self.sql_tool_enabled:
            if not self.readonly_database_url:
                raise ValueError(
                    "READONLY_DATABASE_URL must be set when SQL_TOOL_ENABLED=true "
                    f"and APP_ENV={self.app_env}."
                )
            if self.readonly_database_url == self.database_url:
                raise ValueError(
                    "READONLY_DATABASE_URL must not be the application login - "
                    "it exists so the SQL tool has no write privilege at all."
                )

        if self.jwt_secret_key == DEV_JWT_SECRET:
            raise ValueError(
                f"JWT_SECRET_KEY must be set to a real secret when APP_ENV={self.app_env}. "
                f"Generate one with: python -c 'import secrets;print(secrets.token_urlsafe(48))'"
            )
        if len(self.jwt_secret_key) < 32:
            raise ValueError("JWT_SECRET_KEY must be at least 32 characters.")
        if self.debug:
            raise ValueError(f"DEBUG must be false when APP_ENV={self.app_env}.")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton (cached; cleared in tests)."""
    return Settings()


def generate_secret() -> str:
    return secrets.token_urlsafe(48)
