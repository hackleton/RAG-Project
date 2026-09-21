"""Password hashing, token encoding and configuration guard rails."""

from __future__ import annotations

from datetime import timedelta

import jwt
import pytest

from app.core.config import DEV_JWT_SECRET, AppEnv, Settings
from app.core.errors import AuthenticationError
from app.core.security import (
    BCRYPT_MAX_BYTES,
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)

ROUNDS = 4


@pytest.fixture
def settings() -> Settings:
    return Settings(
        app_env=AppEnv.CI,
        jwt_secret_key="unit-test-secret-padded-to-a-realistic-length",
        database_url="sqlite+aiosqlite:///:memory:",
    )


def test_hashing_is_salted() -> None:
    """Two users with the same password must not share a hash."""
    first = hash_password("same-password", rounds=ROUNDS)
    second = hash_password("same-password", rounds=ROUNDS)

    assert first != second
    assert verify_password("same-password", first)
    assert verify_password("same-password", second)


def test_wrong_password_fails() -> None:
    assert verify_password("nope", hash_password("secret", rounds=ROUNDS)) is False


def test_malformed_hash_fails_closed() -> None:
    """A corrupt row is a failed login, not a 500."""
    assert verify_password("secret", "not-a-bcrypt-hash") is False


def test_oversized_password_is_rejected_rather_than_truncated() -> None:
    """bcrypt silently ignores bytes past 72; accepting them would be a trap."""
    too_long = "a" * (BCRYPT_MAX_BYTES + 1)

    with pytest.raises(ValueError):
        hash_password(too_long, rounds=ROUNDS)

    assert verify_password(too_long, hash_password("a" * BCRYPT_MAX_BYTES, rounds=ROUNDS)) is False


def test_token_round_trip_carries_subject_and_role(settings: Settings) -> None:
    token, expires_in = create_access_token(subject="user-1", role="clinician", settings=settings)

    payload = decode_access_token(token, settings)

    assert payload["sub"] == "user-1"
    assert payload["role"] == "clinician"
    assert payload["type"] == "access"
    assert payload["jti"]
    assert expires_in == settings.access_token_expire_minutes * 60


def test_expired_token_is_rejected(settings: Settings) -> None:
    token, _ = create_access_token(
        subject="user-1",
        role="admin",
        settings=settings,
        expires_delta=timedelta(seconds=-1),
    )

    with pytest.raises(AuthenticationError, match="expired"):
        decode_access_token(token, settings)


def test_token_signed_with_another_key_is_rejected(settings: Settings) -> None:
    token, _ = create_access_token(
        subject="user-1",
        role="admin",
        settings=settings.model_copy(
            update={"jwt_secret_key": "attacker-key-padded-to-a-realistic-length"}
        ),
    )

    with pytest.raises(AuthenticationError):
        decode_access_token(token, settings)


def test_unsigned_token_is_rejected(settings: Settings) -> None:
    """The classic alg=none downgrade must not be accepted."""
    token = jwt.encode({"sub": "user-1", "type": "access"}, key="", algorithm="none")

    with pytest.raises(AuthenticationError):
        decode_access_token(token, settings)


def test_token_without_an_expiry_is_rejected(settings: Settings) -> None:
    token = jwt.encode(
        {"sub": "user-1", "type": "access"},
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )

    with pytest.raises(AuthenticationError):
        decode_access_token(token, settings)


# --- Configuration guard rails ---------------------------------------------


def test_local_environment_may_use_the_development_secret() -> None:
    assert Settings(app_env=AppEnv.LOCAL).jwt_secret_key == DEV_JWT_SECRET


def test_production_refuses_the_development_secret() -> None:
    with pytest.raises(ValueError, match="JWT_SECRET_KEY"):
        Settings(app_env=AppEnv.PRODUCTION, jwt_secret_key=DEV_JWT_SECRET)


def test_production_refuses_a_short_secret() -> None:
    with pytest.raises(ValueError, match="at least 32"):
        Settings(app_env=AppEnv.PRODUCTION, jwt_secret_key="too-short")


def test_production_refuses_debug_mode() -> None:
    with pytest.raises(ValueError, match="DEBUG"):
        Settings(app_env=AppEnv.PRODUCTION, jwt_secret_key="x" * 48, debug=True)


def test_cors_origins_accept_a_comma_separated_string() -> None:
    """Env vars are strings; a list-typed setting has to cope with that."""
    settings = Settings(cors_origins="http://a.test, http://b.test")

    assert settings.cors_origins == ["http://a.test", "http://b.test"]
