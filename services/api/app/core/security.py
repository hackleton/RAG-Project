"""Password hashing and JWT issuing/verification.

bcrypt is used directly rather than through passlib: passlib's bcrypt backend
is unmaintained against bcrypt>=4 and emits a version-detection warning on
import. bcrypt caps input at 72 bytes and silently truncates beyond that, so
longer passwords are rejected explicitly instead.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from functools import partial
from typing import Any, Literal

import anyio.to_thread
import bcrypt
import jwt

from app.core.config import Settings
from app.core.errors import AuthenticationError

BCRYPT_MAX_BYTES = 72
TokenType = Literal["access"]


def hash_password(password: str, *, rounds: int = 12) -> str:
    _reject_oversized(password)
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt(rounds=rounds)).decode()


def verify_password(password: str, password_hash: str) -> bool:
    try:
        _reject_oversized(password)
    except ValueError:
        return False
    try:
        return bcrypt.checkpw(password.encode(), password_hash.encode())
    except ValueError:
        # Malformed hash in the database - treat as a failed login, not a 500.
        return False


async def hash_password_async(password: str, *, rounds: int = 12) -> str:
    """bcrypt at cost 12 takes ~0.25s; keep it off the event loop."""
    return await anyio.to_thread.run_sync(partial(hash_password, password, rounds=rounds))


async def verify_password_async(password: str, password_hash: str) -> bool:
    return await anyio.to_thread.run_sync(verify_password, password, password_hash)


def _reject_oversized(password: str) -> None:
    if len(password.encode()) > BCRYPT_MAX_BYTES:
        raise ValueError(f"Password exceeds {BCRYPT_MAX_BYTES} bytes.")


def create_access_token(
    *,
    subject: str,
    role: str,
    settings: Settings,
    expires_delta: timedelta | None = None,
) -> tuple[str, int]:
    """Return (encoded_jwt, expires_in_seconds)."""
    expires_delta = expires_delta or timedelta(minutes=settings.access_token_expire_minutes)
    now = datetime.now(UTC)
    expire = now + expires_delta
    payload: dict[str, Any] = {
        "sub": subject,
        "role": role,
        "type": "access",
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
        "jti": str(uuid.uuid4()),  # gives Sprint 6 a handle for revocation
    }
    token = jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)
    return token, int(expires_delta.total_seconds())


def decode_access_token(token: str, settings: Settings) -> dict[str, Any]:
    """Decode and validate a bearer token, or raise AuthenticationError."""
    try:
        payload: dict[str, Any] = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
            options={"require": ["exp", "sub", "type"]},
        )
    except jwt.ExpiredSignatureError as exc:
        raise AuthenticationError("Access token has expired.") from exc
    except jwt.InvalidTokenError as exc:
        raise AuthenticationError("Invalid access token.") from exc

    if payload.get("type") != "access":
        raise AuthenticationError("Invalid token type.")
    return payload
