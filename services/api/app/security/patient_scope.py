"""Turning a patient identifier from a request into a patient the caller may see.

Every endpoint that accepts `patient_external_id` goes through `resolve_patient`.
Before Sprint 8 there were four copies of this logic, and between them they
leaked three different ways:

1. **Existence.** An unknown ID answered 404 "Unknown patient" and a real but
   ungranted one answered 403, so comparing the two enumerated which patients
   exist. The document list did the same with an empty 200 against a 403.
2. **The primary key.** The 403 carried the patient's internal UUID in its
   `details`, in a function whose comment said it did not reveal whether the
   patient exists.
3. **The attempt.** Neither refusal was audited, though "someone asked for a
   patient they are not assigned to" is the most important event a clinical
   audit trail can hold.

Now every case gets the same 403 with the same message and no details, the
same answer the tool gate in `app.agents.registry` has always given. The
*reason* (unknown or not granted) goes to the audit row, which only an
administrator can read, so a probe for IDs that don't exist is visible as a
probe.

Found while writing the failure analysis (docs/guide/3-when-it-breaks.md), by
checking whether the demo's refusals appeared in the audit trail.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import PermissionDeniedError
from app.models.enums import AuditAction
from app.models.patient import Patient
from app.security.audit import record_audit_independent
from app.security.principal import Principal

# One message for "no such patient" and for "not your patient". A caller who
# can tell the two apart can enumerate which patients exist.
NO_SUCH_PATIENT = "No accessible patient matches that identifier."


async def resolve_patient(
    session: AsyncSession,
    principal: Principal,
    external_id: str,
    *,
    surface: str,
) -> Patient:
    """The patient this caller may see, or a refusal that says nothing more.

    `surface` names the endpoint in the audit row (`chat`, `agent`, `upload`,
    ...), so a reviewer can see *where* a refused request came in.

    The audit row is written in its own transaction, because raising rolls
    back the request's transaction, and that would erase the record of the
    refusal along with it.
    """
    patient = await session.scalar(select(Patient).where(Patient.external_id == external_id))
    if patient is not None and principal.can_access_patient(patient.id):
        return patient

    await record_audit_independent(
        action=AuditAction.ACCESS_DENIED,
        principal=principal,
        resource_type="patient",
        # Every endpoint bounds it at 64. Capped here too, so a caller that
        # skipped validation cannot lose the audit row to a column overflow.
        resource_id=external_id[:128],
        allowed=False,
        details={"surface": surface, "reason": "unknown" if patient is None else "not_granted"},
    )
    raise PermissionDeniedError(NO_SUCH_PATIENT)
