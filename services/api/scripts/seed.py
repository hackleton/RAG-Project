"""Seed local development data.

Idempotent: re-running updates the demo accounts rather than failing on the
unique email constraint, so it is safe to wire into container start-up.

Refuses to run outside local/CI. Demo passwords come from the environment and
are deliberately weak - this script exists to make a laptop usable, not to
provision anything real.

    docker compose run --rm api python scripts/seed.py
"""

from __future__ import annotations

import asyncio
import os
import random
import sys
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.db import dispose_engine, get_sessionmaker, init_engine
from app.core.security import hash_password
from app.models import LabResult, Patient, PatientAccessGrant, Role, User

# (env var prefix, name, role)
DEMO_USERS = [
    ("SEED_ADMIN", "Avery Admin", Role.ADMIN),
    ("SEED_CLINICIAN", "Casey Clinician", Role.CLINICIAN),
    ("SEED_ANALYST", "Alex Analyst", Role.ANALYST),
]

DEMO_PATIENTS = [
    ("SYN-0001", "Synthetic Patient 0001", 1958, "female"),
    ("SYN-0002", "Synthetic Patient 0002", 1972, "male"),
    ("SYN-0003", "Synthetic Patient 0003", 1985, "female"),
]

# (test_name, code, unit, reference low/high, plausible value range)
LAB_PANEL = [
    ("Hemoglobin A1c", "4548-4", "%", 4.0, 5.6, (5.2, 9.4)),
    ("Fasting Glucose", "1558-6", "mg/dL", 70.0, 99.0, (78.0, 165.0)),
    ("Creatinine", "2160-0", "mg/dL", 0.6, 1.3, (0.6, 1.9)),
    ("LDL Cholesterol", "2089-1", "mg/dL", 0.0, 100.0, (70.0, 190.0)),
    ("Systolic Blood Pressure", "8480-6", "mmHg", 90.0, 120.0, (108.0, 168.0)),
]

MONTHLY_READINGS = 8


async def seed(session: AsyncSession, rounds: int) -> None:
    users = await _upsert_users(session, rounds)
    patients = await _upsert_patients(session)
    await session.flush()

    # The clinician sees the first two patients; the third is deliberately out
    # of scope so the access-control tests and the demo have a negative case.
    await _grant(session, users[Role.CLINICIAN], patients[:2])
    await _seed_labs(session, patients)
    await session.commit()

    print(f"Seeded {len(users)} users, {len(patients)} patients.")
    for prefix, _, role in DEMO_USERS:
        print(f"  {role.value:<10} {os.environ[f'{prefix}_EMAIL']}")


async def _upsert_users(session: AsyncSession, rounds: int) -> dict[Role, User]:
    result: dict[Role, User] = {}
    for prefix, name, role in DEMO_USERS:
        email = os.environ[f"{prefix}_EMAIL"].strip().lower()
        password = os.environ[f"{prefix}_PASSWORD"]

        user = await session.scalar(select(User).where(User.email == email))
        if user is None:
            user = User(email=email, name=name, role=role)
            session.add(user)
        user.name = name
        user.role = role
        user.is_active = True
        user.hashed_password = hash_password(password, rounds=rounds)
        result[role] = user
    return result


async def _upsert_patients(session: AsyncSession) -> list[Patient]:
    patients: list[Patient] = []
    for external_id, display_name, birth_year, sex in DEMO_PATIENTS:
        patient = await session.scalar(select(Patient).where(Patient.external_id == external_id))
        if patient is None:
            patient = Patient(external_id=external_id)
            session.add(patient)
        patient.display_name = display_name
        patient.birth_year = birth_year
        patient.sex = sex
        patient.is_synthetic = True
        patients.append(patient)
    return patients


async def _grant(session: AsyncSession, user: User, patients: list[Patient]) -> None:
    existing = set(
        await session.scalars(
            select(PatientAccessGrant.patient_id).where(PatientAccessGrant.user_id == user.id)
        )
    )
    for patient in patients:
        if patient.id not in existing:
            session.add(PatientAccessGrant(user_id=user.id, patient_id=patient.id))


async def _seed_labs(session: AsyncSession, patients: list[Patient]) -> None:
    already_seeded = await session.scalar(select(LabResult.id).limit(1))
    if already_seeded is not None:
        return

    # Fixed seed: the evaluation dataset references these values, so they must
    # be reproducible across machines and re-runs.
    rng = random.Random(20260922)
    now = datetime.now(UTC)

    for patient in patients:
        for test_name, code, unit, low, high, (min_v, max_v) in LAB_PANEL:
            for month in range(MONTHLY_READINGS):
                value = round(rng.uniform(min_v, max_v), 1)
                session.add(
                    LabResult(
                        patient_id=patient.id,
                        test_name=test_name,
                        test_code=code,
                        value=value,
                        unit=unit,
                        reference_low=low,
                        reference_high=high,
                        abnormal_flag=_flag(value, low, high),
                        collected_at=now - timedelta(days=30 * (MONTHLY_READINGS - month)),
                    )
                )


def _flag(value: float, low: float, high: float) -> str | None:
    if value < low:
        return "L"
    if value > high:
        return "H"
    return None


async def main() -> int:
    settings = get_settings()
    if not settings.is_local:
        print(
            f"Refusing to seed demo accounts with APP_ENV={settings.app_env.value}.",
            file=sys.stderr,
        )
        return 1

    missing = [
        f"{prefix}_{suffix}"
        for prefix, _, _ in DEMO_USERS
        for suffix in ("EMAIL", "PASSWORD")
        if not os.environ.get(f"{prefix}_{suffix}")
    ]
    if missing:
        print(f"Missing environment variables: {', '.join(missing)}", file=sys.stderr)
        return 1

    init_engine(settings)
    try:
        async with get_sessionmaker()() as session:
            await seed(session, settings.bcrypt_rounds)
    finally:
        await dispose_engine()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
