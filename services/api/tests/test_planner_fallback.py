"""The model-driven planner falls back to the rules when the model lets it down.

Two ways it can: the model answers with something that isn't a plan, or the
model call fails - unreachable, or out of tokens before it wrote anything,
which is what a reasoning model at its default "high" effort did here. Before
this, only the first was handled, and neither had a test.
"""

from __future__ import annotations

from app.agents.planner import LlmPlanner, RuleBasedPlanner
from app.agents.registry import ToolRegistry
from app.core.errors import ServiceUnavailableError
from app.models import Patient, User
from app.providers.llm import Completion
from tests.conftest import principal_for

QUESTION = "What is the HbA1c trend for this patient?"


class FailingModel:
    model = "fails"

    async def complete(self, *, system: str, user: str) -> Completion:
        raise ServiceUnavailableError("The model used its whole output allowance.")


class RamblingModel:
    model = "rambles"

    async def complete(self, *, system: str, user: str) -> Completion:
        return Completion(text="I would probably look at the labs, maybe.", model=self.model)


async def plan(planner: object, registry: ToolRegistry, user: User, patients: list[Patient]):
    principal = principal_for(user, patients)
    return await planner.plan(  # type: ignore[attr-defined]
        QUESTION,
        principal,
        patient_external_id="SYN-0001",
        available=registry.available_to(principal),
        max_calls=4,
    )


async def test_a_failing_model_falls_back_to_the_rules(
    registry: ToolRegistry, users: dict[str, User], patients: list[Patient]
) -> None:
    by_rules = await plan(RuleBasedPlanner(), registry, users["clinician"], patients)
    fallen_back = await plan(
        LlmPlanner(FailingModel(), registry), registry, users["clinician"], patients
    )

    assert [call.name for call in fallen_back] == [call.name for call in by_rules]
    assert fallen_back, "the rules route a lab question somewhere"


async def test_an_unreadable_plan_falls_back_to_the_rules(
    registry: ToolRegistry, users: dict[str, User], patients: list[Patient]
) -> None:
    by_rules = await plan(RuleBasedPlanner(), registry, users["clinician"], patients)
    fallen_back = await plan(
        LlmPlanner(RamblingModel(), registry), registry, users["clinician"], patients
    )

    assert [call.name for call in fallen_back] == [call.name for call in by_rules]
