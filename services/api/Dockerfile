# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Microsoft ODBC Driver 18: pyodbc needs it to reach SQL Server. pyodbc itself
# ships manylinux wheels, so only the runtime driver is required - no compiler.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl gnupg ca-certificates \
    && curl -fsSL https://packages.microsoft.com/keys/microsoft.asc \
        | gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg \
    && printf '%s\n' \
        "deb [arch=amd64,arm64 signed-by=/usr/share/keyrings/microsoft-prod.gpg] https://packages.microsoft.com/debian/12/prod bookworm main" \
        > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y --no-install-recommends msodbcsql18 unixodbc \
    && apt-get purge -y --auto-remove gnupg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

# Dependency layer: only invalidated when pyproject.toml changes.
COPY pyproject.toml ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel && \
    pip install "fastapi>=0.115.0" "uvicorn[standard]>=0.32.0" "pydantic[email]>=2.9.0" \
        "pydantic-settings>=2.6.0" "sqlalchemy[asyncio]>=2.0.36" \
        "aioodbc>=0.5.0" "pyodbc>=5.2.0" \
        "alembic>=1.14.0" "pyjwt>=2.9.0" "bcrypt>=4.2.0" "python-multipart>=0.0.12" \
        "structlog>=24.4.0" "redis>=5.2.0" "pypdf>=5.1.0" "python-docx>=1.1.2" \
        "qdrant-client>=1.12.0" "fastembed>=0.5.0" "httpx>=0.27.2"

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
