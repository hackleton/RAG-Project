"""The tool registry and the authorisation gate.

**The router is untrusted.** Whatever chooses a tool - a language model, an
intent rule, or a sentence embedded in a retrieved document that the model
decided to follow - produces nothing more than a name and a bag of arguments.
Treating that as an authorisation decision would make the planner a security
boundary, and a text generator cannot be one.

So every call, from every source, passes the same gate in the same order:

1. **Does the tool exist?** An unknown name is denied and audited, never
   raised - a planner naming a tool that was never built is a routine event,
   and the interesting part is the record that it tried.
2. **May this role use it?** From the registry, which took it from the role,
   not from anything in the request.
3. **Do the arguments type-check?** Strictly, with extra fields forbidden, so
   an argument the tool does not understand is a refusal rather than a value
   silently ignored.
4. **May the caller see this patient?** Resolved once, here, so no individual
   tool can forget to do it.
5. Only then does the tool run, and the outcome is audited either way.

Project plan section 11 asks for "tool authorization before every privileged
tool call" and section 12's "call an unauthorized tool" test case lands
squarely on step 2.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.errors import PermissionDeniedError
from app.core.logging import get_logger
from app.models.enums import AuditAction, Role
from app.models.patient import Patient
from app.security.audit import record_audit
from app.security.principal import Principal
from app.security.rbac import assert_patient_access
from app.tools.base import Tool, ToolContext, ToolResult

logger = get_logger(__name__)

# One message for "no such patient" and for "not your patient". A caller who
# can tell the two apart can enumerate which patients exist.
NO_SUCH_PATIENT = "No accessible patient matches that identifier."


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    arguments: type[BaseModel]
    allowed_roles: frozenset[Role]
    tool: Tool
    # Tools whose use is itself the auditable event, beyond the generic call
    # record - a generated report is one (plan section 11).
    audit_action: AuditAction = AuditAction.TOOL_CALLED


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A proposed call. Nothing here has been checked yet."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    # Why the planner chose it. Shown in the trace; never acted on.
    reason: str | None = None


@dataclass(slots=True)
class DispatchOutcome:
    call: ToolCall
    allowed: bool
    result: ToolResult | None = None
    denial: str | None = None
    # Stable machine-readable denial reason, for tests and dashboards.
    denial_code: str | None = None
    duration_ms: int = 0


class ToolRegistry:
    def __init__(self, specs: list[ToolSpec]) -> None:
        self._specs = {spec.name: spec for spec in specs}

    def register(self, spec: ToolSpec) -> None:
        """Add a tool after construction.

        Exists for the composite tools, which need the dispatcher and so
        cannot be built until the registry they will dispatch through is.
        """
        self._specs[spec.name] = spec

    @property
    def names(self) -> list[str]:
        return sorted(self._specs)

    def available_to(self, principal: Principal) -> list[str]:
        """Tool names this role may use - what the planner may choose from."""
        return sorted(
            name for name, spec in self._specs.items() if principal.role in spec.allowed_roles
        )

    def catalogue(self, principal: Principal) -> list[dict[str, Any]]:
        """The tools this caller may use, as the planner sees them.

        Filtered by role, so a model is not shown a tool it would only be
        denied - and so the catalogue cannot become a list of capabilities to
        enumerate. The filter is a convenience, not the control: the gate
        re-checks the role even for a tool it never advertised.
        """
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "arguments": _argument_summary(spec.arguments),
            }
            for spec in sorted(self._specs.values(), key=lambda item: item.name)
            if principal.role in spec.allowed_roles
        ]

    async def dispatch(
        self,
        session: AsyncSession,
        principal: Principal,
        settings: Settings,
        call: ToolCall,
        *,
        default_patient_external_id: str | None = None,
    ) -> DispatchOutcome:
        started = time.perf_counter()

        spec = self._specs.get(call.name)
        if spec is None:
            return await self._deny(
                session, principal, call, "unknown_tool", f"No tool named '{call.name}'.", started
            )

        if principal.role not in spec.allowed_roles:
            return await self._deny(
                session,
                principal,
                call,
                "role_not_permitted",
                f"The {principal.role.value} role may not use '{spec.name}'.",
                started,
            )

        try:
            arguments = spec.arguments.model_validate(call.arguments)
        except ValidationError as invalid:
            return await self._deny(
                session,
                principal,
                call,
                "invalid_arguments",
                _first_error(invalid),
                started,
            )

        external_id = getattr(arguments, "patient_external_id", None) or default_patient_external_id
        patient_id: uuid.UUID | None = None
        if external_id is not None:
            try:
                patient_id = await self._resolve_patient(session, principal, external_id)
            except PermissionDeniedError:
                return await self._deny(
                    session, principal, call, "patient_not_permitted", NO_SUCH_PATIENT, started
                )
            if patient_id is None:
                return await self._deny(
                    session, principal, call, "unknown_patient", NO_SUCH_PATIENT, started
                )

        context = ToolContext(
            session=session,
            principal=principal,
            settings=settings,
            patient_id=patient_id,
            patient_external_id=external_id,
        )
        result = await spec.tool.run(context, arguments)
        duration_ms = int((time.perf_counter() - started) * 1000)

        await record_audit(
            session,
            action=spec.audit_action,
            principal=principal,
            resource_type="tool",
            resource_id=spec.name,
            allowed=True,
            details={
                "patient_external_id": external_id,
                "summary": result.summary,
                "evidence": len(result.evidence),
                "duration_ms": duration_ms,
            },
        )
        logger.info(
            "tool_called",
            tool=spec.name,
            evidence=len(result.evidence),
            duration_ms=duration_ms,
        )
        return DispatchOutcome(call=call, allowed=True, result=result, duration_ms=duration_ms)

    # --- internals --------------------------------------------------------

    async def _resolve_patient(
        self, session: AsyncSession, principal: Principal, external_id: str
    ) -> uuid.UUID | None:
        patient = await session.scalar(select(Patient).where(Patient.external_id == external_id))
        if patient is None:
            return None
        assert_patient_access(principal, patient.id)
        return patient.id

    async def _deny(
        self,
        session: AsyncSession,
        principal: Principal,
        call: ToolCall,
        code: str,
        message: str,
        started: float,
    ) -> DispatchOutcome:
        await record_audit(
            session,
            action=AuditAction.TOOL_DENIED,
            principal=principal,
            resource_type="tool",
            resource_id=call.name,
            allowed=False,
            details={"reason": code, "message": message},
        )
        logger.warning("tool_denied", tool=call.name, reason=code)
        return DispatchOutcome(
            call=call,
            allowed=False,
            denial=message,
            denial_code=code,
            duration_ms=int((time.perf_counter() - started) * 1000),
        )


def _argument_summary(model: type[BaseModel]) -> dict[str, Any]:
    """A compact argument description for the planner prompt.

    The full JSON schema would be several hundred tokens per tool and is mostly
    boilerplate the model does not need.
    """
    return {
        name: {
            "type": _type_name(field_info.annotation),
            "required": field_info.is_required(),
            **({"description": field_info.description} if field_info.description else {}),
        }
        for name, field_info in model.model_fields.items()
    }


def _type_name(annotation: Any) -> str:
    name = getattr(annotation, "__name__", None)
    return name if isinstance(name, str) else "string"


def _first_error(invalid: ValidationError) -> str:
    error = invalid.errors()[0]
    location = ".".join(str(part) for part in error["loc"]) or "arguments"
    return f"Invalid argument '{location}': {error['msg']}."
