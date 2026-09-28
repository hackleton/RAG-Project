"""Role, patient-scope and sensitivity checks.

Pure predicates with no HTTP or framework coupling, so they can be called from
request handlers, agent tool dispatch and the SQL guard alike, and unit-tested
without a client. The FastAPI wrapper lives in `app.api.deps.require_roles`.

These are application controls, deliberately independent of the prompt: an
unauthorised request is refused here whether it came from a user, an agent tool
call, or text embedded in a retrieved document (project plan section 12).
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable

from app.core.errors import PermissionDeniedError
from app.models.enums import AccessLevel, Role
from app.security.principal import Principal


def assert_role(principal: Principal, allowed: Iterable[Role]) -> None:
    allowed_set = frozenset(allowed)
    if principal.role not in allowed_set:
        raise PermissionDeniedError(
            "This action requires a different role.",
            details={
                "required_roles": sorted(r.value for r in allowed_set),
                "actual_role": principal.role.value,
            },
        )


def assert_patient_access(principal: Principal, patient_id: uuid.UUID | None) -> None:
    if not principal.can_access_patient(patient_id):
        # No details. This used to return the patient's internal UUID, which
        # confirmed the patient existed and handed over its primary key to a
        # caller who was being refused it. See `app.security.patient_scope`.
        raise PermissionDeniedError("You do not have access to this patient's data.")


def assert_access_level(principal: Principal, level: AccessLevel) -> None:
    if not principal.can_access_level(level):
        raise PermissionDeniedError(
            "You do not have access to this document's sensitivity level.",
            details={"access_level": level.value},
        )
