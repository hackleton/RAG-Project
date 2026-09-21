"""Application settings.

All configuration comes from the environment (12-factor). Nothing in this file
may carry a production-usable default: `Settings.validate_production_safety`
refuses to boot a non-development environment that is still using dev values.
"""

from __future__ import annotations

import json
import secrets
from enum import StrEnum
from functools import lru_cache
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

    `.env.example` lists the Azure keys with empty values so the file documents
    them; an empty string must mean "not configured", not "invalid URL".
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
    # asyncpg driver is required; the sync driver is only used by Alembic.
    database_url: str = "postgresql+asyncpg://copilot:copilot@localhost:5432/copilot"
    database_pool_size: int = 5
    database_max_overflow: int = 10
    database_echo: bool = False
    redis_url: str = "redis://localhost:6379/0"

    # --- Auth ------------------------------------------------------------
    jwt_secret_key: str = DEV_JWT_SECRET
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    # bcrypt cost. 12 is the production floor; tests override it to keep the
    # suite fast (bcrypt is intentionally slow).
    bcrypt_rounds: int = 12

    # --- Azure (wired up in later sprints; optional so Sprint 1 boots clean)
    azure_openai_endpoint: OptionalUrl = None
    azure_openai_api_key: OptionalStr = None
    azure_openai_chat_deployment: OptionalStr = None
    azure_openai_embedding_deployment: OptionalStr = None
    azure_search_endpoint: OptionalUrl = None
    azure_search_api_key: OptionalStr = None
    azure_search_index_name: str = "healthcare-chunks"
    azure_document_intelligence_endpoint: OptionalUrl = None
    azure_document_intelligence_api_key: OptionalStr = None
    azure_storage_connection_string: OptionalStr = None
    azure_storage_container: str = "documents"

    @property
    def is_local(self) -> bool:
        return self.app_env in (AppEnv.LOCAL, AppEnv.CI)

    @model_validator(mode="after")
    def validate_production_safety(self) -> Settings:
        if self.is_local:
            return self
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
