"""Authentication payloads."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from app.models.enums import Role


class LoginRequest(BaseModel):
    """JSON login. The OAuth2 form flow is also accepted at /auth/token."""

    email: EmailStr
    password: str = Field(min_length=1, max_length=72)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = Field(description="Token lifetime in seconds.")


class UserRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    email: EmailStr
    name: str
    role: Role
    is_active: bool
    created_at: datetime


class CurrentUserResponse(UserRead):
    """`/auth/me` also returns the resolved patient scope used for filtering."""

    accessible_patient_ids: list[uuid.UUID] | None = Field(
        default=None,
        description="Null means unrestricted (admin); a list means filtered access.",
    )


class LogoutResponse(BaseModel):
    """Whether the presented token was added to the revocation list.

    `false` means the caller authenticated some other way than by presenting a
    token this endpoint could read - the cookie is cleared either way, and a
    client should treat the session as over regardless.
    """

    revoked: bool
