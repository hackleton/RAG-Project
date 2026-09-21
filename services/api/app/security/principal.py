"""The authenticated caller, resolved once per request.

Everything downstream - retrieval filters, tool authorisation, SQL row scoping,
audit records - reads the caller's scope from this object rather than
re-querying or, worse, trusting a value from the request body.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from app.models.enums import ACCESS_LEVEL_ROLES, AccessLevel, Role


@dataclass(frozen=True, slots=True)
class Principal:
    id: uuid.UUID
    email: str
    name: str
    role: Role
    # None means "every patient" (admin). An empty frozenset means "none".
    accessible_patient_ids: frozenset[uuid.UUID] | None = None
    request_id: str | None = field(default=None, compare=False)

    @property
    def is_admin(self) -> bool:
        return self.role is Role.ADMIN

    @property
    def visible_access_levels(self) -> frozenset[AccessLevel]:
        return frozenset(level for level, roles in ACCESS_LEVEL_ROLES.items() if self.role in roles)

    def can_access_patient(self, patient_id: uuid.UUID | None) -> bool:
        if patient_id is None:
            # Documents with no patient link are non-patient material (guidelines).
            return True
        if self.accessible_patient_ids is None:
            return True
        return patient_id in self.accessible_patient_ids

    def can_access_level(self, level: AccessLevel) -> bool:
        return self.role in ACCESS_LEVEL_ROLES[level]
