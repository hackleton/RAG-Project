# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /srv

# Dependency layer: only invalidated when pyproject.toml changes.
COPY pyproject.toml ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel && \
    pip install "fastapi>=0.115.0" "uvicorn[standard]>=0.32.0" "pydantic[email]>=2.9.0" \
        "pydantic-settings>=2.6.0" "sqlalchemy[asyncio]>=2.0.36" "asyncpg>=0.30.0" \
        "alembic>=1.14.0" "pyjwt>=2.9.0" "bcrypt>=4.2.0" "python-multipart>=0.0.12" \
        "structlog>=24.4.0" "redis>=5.2.0"

FROM base AS dev
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "pytest>=8.3.0" "pytest-asyncio>=0.24.0" "httpx>=0.27.2" \
        "aiosqlite>=0.20.0" "ruff>=0.7.0" "mypy>=1.13.0"
COPY . .
RUN pip install --no-deps -e .
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

FROM base AS runtime
COPY . .
RUN pip install --no-deps . && \
    useradd --system --uid 10001 --no-create-home appuser && \
    chown -R appuser:appuser /srv
USER appuser
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
