"""Test fixtures.

The suite runs against in-memory SQLite rather than Postgres so `pytest` needs
no running container and stays fast enough to sit in a pre-commit loop. The
schema is built from the same `Base.metadata` the migrations are generated
from; migrations themselves are exercised separately against real Postgres
(`docker compose run --rm api alembic upgrade head`), because SQLite cannot
represent JSONB or the read-only role.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import db as db_module
from app.core.config import AppEnv, Settings
from app.core.security import hash_password
from app.main import create_app
from app.models import Base, Patient, PatientAccessGrant, Role, User

# bcrypt at the production cost would add ~0.25s to every login test.
TEST_BCRYPT_ROUNDS = 4

ADMIN_PASSWORD = "admin-password"
CLINICIAN_PASSWORD = "clinician-password"
ANALYST_PASSWORD = "analyst-password"


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings(
        app_env=AppEnv.CI,
        database_url="sqlite+aiosqlite:///:memory:",
        jwt_secret_key="test-secret-not-used-outside-the-suite-0123456789",
        log_level="WARNING",
        log_format="console",
        bcrypt_rounds=TEST_BCRYPT_ROUNDS,
    )


@pytest.fixture
async def app(settings: Settings) -> AsyncIterator[object]:
    application = create_app(settings)
    engine = db_module.init_engine(settings)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    try:
        yield application
    finally:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.drop_all)
        await db_module.dispose_engine()


@pytest.fixture
async def session(app: object) -> AsyncIterator[AsyncSession]:
    """A session on the same engine the app uses, for arranging fixtures."""
    async with db_module.get_sessionmaker()() as db_session:
        yield db_session


@pytest.fixture
async def client(app: object) -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=app)  # type: ignore[arg-type]
    # `lifespan` is not run by ASGITransport; the app fixture already built the
    # engine, which is the only thing startup does.
    async with AsyncClient(transport=transport, base_url="http://test") as async_client:
        yield async_client


@pytest.fixture
def api(settings: Settings) -> str:
    return settings.api_prefix


@pytest.fixture
async def patients(session: AsyncSession) -> list[Patient]:
    rows = [
        Patient(external_id="SYN-0001", display_name="Synthetic Patient 0001", birth_year=1958),
        Patient(external_id="SYN-0002", display_name="Synthetic Patient 0002", birth_year=1972),
    ]
    session.add_all(rows)
    await session.commit()
    for row in rows:
        await session.refresh(row)
    return rows


@pytest.fixture
async def users(session: AsyncSession, patients: list[Patient]) -> dict[str, User]:
    """One user per role. The clinician is scoped to the first patient only."""
    created = {
        "admin": User(
            email="admin@example.com",
            name="Admin User",
            role=Role.ADMIN,
            hashed_password=hash_password(ADMIN_PASSWORD, rounds=TEST_BCRYPT_ROUNDS),
        ),
        "clinician": User(
            email="clinician@example.com",
            name="Clinician User",
            role=Role.CLINICIAN,
            hashed_password=hash_password(CLINICIAN_PASSWORD, rounds=TEST_BCRYPT_ROUNDS),
        ),
        "analyst": User(
            email="analyst@example.com",
            name="Analyst User",
            role=Role.ANALYST,
            hashed_password=hash_password(ANALYST_PASSWORD, rounds=TEST_BCRYPT_ROUNDS),
        ),
    }
    session.add_all(created.values())
    await session.flush()
    session.add(PatientAccessGrant(user_id=created["clinician"].id, patient_id=patients[0].id))
    await session.commit()
    for user in created.values():
        await session.refresh(user)
    return created


async def login(client: AsyncClient, api: str, email: str, password: str) -> str:
    response = await client.post(f"{api}/auth/login", json={"email": email, "password": password})
    assert response.status_code == 200, response.text
    token: str = response.json()["access_token"]
    return token


@pytest.fixture
async def admin_token(client: AsyncClient, api: str, users: dict[str, User]) -> str:
    return await login(client, api, users["admin"].email, ADMIN_PASSWORD)


@pytest.fixture
async def clinician_token(client: AsyncClient, api: str, users: dict[str, User]) -> str:
    return await login(client, api, users["clinician"].email, CLINICIAN_PASSWORD)


@pytest.fixture
async def analyst_token(client: AsyncClient, api: str, users: dict[str, User]) -> str:
    return await login(client, api, users["analyst"].email, ANALYST_PASSWORD)


def auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def unknown_id() -> uuid.UUID:
    return uuid.uuid4()
