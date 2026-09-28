"""The cost and latency dashboard, over HTTP."""

from __future__ import annotations

import pytest
from httpx import AsyncClient

from tests.conftest import auth
from tests.test_retrieval import DIABETES_NOTE, upload


async def ask(client: AsyncClient, api: str, token: str, question: str) -> dict:
    response = await client.post(f"{api}/chat", headers=auth(token), json={"question": question})
    assert response.status_code == 200, response.text
    body: dict = response.json()
    return body


async def test_the_dashboard_is_admin_only(
    client: AsyncClient, api: str, clinician_token: str, analyst_token: str
) -> None:
    """A per-user latency and cost breakdown is a view of what colleagues do."""
    for token in (clinician_token, analyst_token):
        response = await client.get(f"{api}/metrics/overview", headers=auth(token))
        assert response.status_code == 403, response.text


async def test_an_answer_shows_up_with_its_cost_and_its_components(
    client: AsyncClient, api: str, admin_token: str, clinician_token: str
) -> None:
    await upload(
        client,
        api,
        clinician_token,
        DIABETES_NOTE,
        "note.txt",
        patient_external_id="SYN-0001",
        access_level="restricted",
    )
    await ask(client, api, clinician_token, "what is the a1c?")

    response = await client.get(f"{api}/metrics/overview", headers=auth(admin_token))
    assert response.status_code == 200, response.text
    body = response.json()

    answers = body["answers"]
    assert answers["answers"] >= 1
    assert answers["sampled"] >= 1
    assert answers["latency_ms"]["p95"] > 0
    assert answers["prompt_tokens"] > 0
    # Local generation is free; the field exists so the accounting is already
    # in place when it is not.
    assert answers["cost_usd"] == 0.0

    assert [model["model"] for model in answers["models"]] == ["echo"]

    components = {component["name"]: component for component in answers["components"]}
    assert "llm.complete" in components
    assert any(name.startswith("retrieval.") for name in components)
    # Self time, so the shares are a real division of the traced total. Each
    # share is rounded to 4dp, so the sum is 1 only to within that rounding -
    # exact equality passed or failed depending on the run's timings.
    shares = [component["share"] for component in answers["components"]]
    assert sum(shares) == pytest.approx(1.0, abs=0.0001 * len(shares))

    assert body["daily"][-1]["answers"] >= 1


async def test_the_cache_block_says_what_kind_of_cache_it_is(
    client: AsyncClient, api: str, admin_token: str, clinician_token: str
) -> None:
    """`shared` false means the login limiter is per-replica again - the
    dashboard has to say so rather than leave it to be inferred."""
    await upload(
        client,
        api,
        clinician_token,
        DIABETES_NOTE,
        "note.txt",
        patient_external_id="SYN-0001",
        access_level="restricted",
    )
    await ask(client, api, clinician_token, "what is the a1c?")

    body = (await client.get(f"{api}/metrics/overview", headers=auth(admin_token))).json()
    cache = body["cache"]

    assert cache["backend"] == "memory"
    assert cache["enabled"] is True
    assert cache["shared"] is False
    assert cache["reachable"] is True
    assert {row["namespace"] for row in cache["namespaces"]} >= {"embedding", "ranking"}


async def test_the_process_block_is_labelled_as_since_start(
    client: AsyncClient, api: str, admin_token: str
) -> None:
    body = (await client.get(f"{api}/metrics/overview", headers=auth(admin_token))).json()
    process = body["process"]

    assert process["started_at"]
    assert process["uptime_seconds"] >= 0
    assert process["requests"]["total"] >= 1
    assert process["requests"]["error_rate"] == 0.0


async def test_the_window_is_bounded(client: AsyncClient, api: str, admin_token: str) -> None:
    assert (
        await client.get(f"{api}/metrics/overview", params={"days": 0}, headers=auth(admin_token))
    ).status_code == 422
    assert (
        await client.get(f"{api}/metrics/overview", params={"days": 400}, headers=auth(admin_token))
    ).status_code == 422
