"""The OpenRouter provider, against a fake network.

`httpx.MockTransport` answers every request, so these run with no key, no
network and no credits. What they pin down is the part that has to be right
before a real call is ever made: what is sent, how an answer and its cost are
read, which failures are retried, and that the key never leaves the
`Authorization` header.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from app.core.config import AppEnv, Settings
from app.core.config import LLMProvider as LLMProviderName
from app.core.errors import ServiceUnavailableError
from app.providers import llm as llm_module
from app.providers.llm import OpenRouterLLM, build_llm_provider

# Deliberately *not* shaped like a real key (no `sk-or-v1-` prefix): secret
# scanners match that prefix, and a fake key that trips one blocks a push and
# teaches people to dismiss the alert that will one day be real.
KEY = "fake-openrouter-key-for-tests-only-0123456789"
MODEL = "anthropic/claude-sonnet-5"


def answer(content: str | None = "Metformin 1000 mg twice daily [1].", **extra: Any) -> dict:
    body: dict[str, Any] = {
        "id": "gen-1",
        "model": MODEL,
        "choices": [
            {"finish_reason": "stop", "message": {"role": "assistant", "content": content}}
        ],
        "usage": {"prompt_tokens": 812, "completion_tokens": 64, "cost": 0.002264},
    }
    body.update(extra)
    return body


class Network:
    """Replies from a script, one response per request, and records what was sent."""

    def __init__(self, *replies: httpx.Response | Exception) -> None:
        self.replies = list(replies)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def provider(network: Network, **options: Any) -> OpenRouterLLM:
    return OpenRouterLLM(
        api_key=KEY,
        model=MODEL,
        backoff_seconds=0.0,
        transport=httpx.MockTransport(network),
        **options,
    )


class Recorder:
    """Stands in for the module logger, so every logged field can be inspected."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def warning(self, event: str, **fields: Any) -> None:
        self.events.append((event, fields))

    info = debug = error = warning


@pytest.fixture
def logs(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    recorder = Recorder()
    monkeypatch.setattr(llm_module, "logger", recorder)
    return recorder


# --- the happy path -----------------------------------------------------------


async def test_the_request_is_an_openai_style_chat_completion() -> None:
    network = Network(httpx.Response(200, json=answer()))
    await provider(network, temperature=0.0, max_tokens=512).complete(
        system="You answer from context.", user="QUESTION: dose?"
    )

    request = network.requests[0]
    assert request.method == "POST"
    assert str(request.url) == "https://openrouter.ai/api/v1/chat/completions"
    assert request.headers["Authorization"] == f"Bearer {KEY}"
    assert json.loads(request.content) == {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You answer from context."},
            {"role": "user", "content": "QUESTION: dose?"},
        ],
        "temperature": 0.0,
        "max_tokens": 512,
    }


async def test_the_answer_tokens_and_real_cost_are_read() -> None:
    completion = await provider(Network(httpx.Response(200, json=answer()))).complete(
        system="s", user="u"
    )

    assert completion.text == "Metformin 1000 mg twice daily [1]."
    assert completion.model == MODEL
    assert (completion.prompt_tokens, completion.completion_tokens) == (812, 64)
    # The first provider in the project with a real price on each call.
    assert completion.estimated_cost_usd == pytest.approx(0.002264)


async def test_a_missing_cost_is_unknown_not_free() -> None:
    body = answer()
    del body["usage"]["cost"]
    completion = await provider(Network(httpx.Response(200, json=body))).complete(
        system="s", user="u"
    )
    assert completion.estimated_cost_usd is None


async def test_empty_content_is_an_empty_answer_not_a_crash() -> None:
    completion = await provider(Network(httpx.Response(200, json=answer(None)))).complete(
        system="s", user="u"
    )
    assert completion.text == ""


async def test_a_truncated_answer_is_logged(logs: Recorder) -> None:
    body = answer()
    body["choices"][0]["finish_reason"] = "length"
    await provider(Network(httpx.Response(200, json=body))).complete(system="s", user="u")
    assert [event for event, _ in logs.events] == ["llm_truncated"]


async def test_one_connection_is_reused_across_calls() -> None:
    network = Network(httpx.Response(200, json=answer()), httpx.Response(200, json=answer()))
    llm = provider(network)
    await llm.complete(system="s", user="u")
    first = llm._client
    await llm.complete(system="s", user="u")
    assert llm._client is first
    await llm.aclose()
    assert llm._client is None


# --- failures -----------------------------------------------------------------


async def test_an_error_inside_a_200_is_an_error_not_an_empty_answer() -> None:
    """OpenRouter documents this: the provider failed after the response began."""
    body = {"error": {"code": 502, "message": "Provider returned error"}}
    network = Network(httpx.Response(200, json=body))
    with pytest.raises(ServiceUnavailableError, match="unavailable"):
        await provider(network, max_retries=0).complete(system="s", user="u")
    assert len(network.requests) == 1


async def test_a_transient_error_inside_a_200_is_retried_like_the_status() -> None:
    """A 502 in the body is the same failure as a 502 status, so it gets the retry."""
    network = Network(
        httpx.Response(200, json={"error": {"code": 502, "message": "Provider returned error"}}),
        httpx.Response(200, json=answer()),
    )
    completion = await provider(network).complete(system="s", user="u")
    assert completion.text.startswith("Metformin")
    assert len(network.requests) == 2


async def test_a_200_with_no_choices_is_an_error() -> None:
    with pytest.raises(ServiceUnavailableError):
        await provider(Network(httpx.Response(200, json={"id": "x"}))).complete(
            system="s", user="u"
        )


@pytest.mark.parametrize(
    ("status", "message", "expected"),
    [
        (401, "No auth credentials found", "rejected the API key"),
        (402, "Insufficient credits", "run out of credits"),
        (403, "Flagged by moderation", "refused the request"),
        (400, "anthropic/claude-nope is not a valid model ID", "does not recognise the model"),
        (400, "messages must not be empty", "rejected the request as invalid"),
    ],
)
async def test_a_refusal_says_what_to_fix_and_is_not_retried(
    status: int, message: str, expected: str
) -> None:
    network = Network(httpx.Response(status, json={"error": {"code": status, "message": message}}))
    with pytest.raises(ServiceUnavailableError, match=expected):
        await provider(network).complete(system="s", user="u")
    # Repeating a bad key or an empty account only repeats the refusal.
    assert len(network.requests) == 1


async def test_a_rate_limit_is_retried_and_can_succeed() -> None:
    network = Network(
        httpx.Response(429, headers={"Retry-After": "0"}, json={"error": {"code": 429}}),
        httpx.Response(200, json=answer()),
    )
    completion = await provider(network).complete(system="s", user="u")
    assert completion.text.startswith("Metformin")
    assert len(network.requests) == 2


async def test_retries_stop_at_the_limit() -> None:
    unavailable = httpx.Response(503, json={"error": {"code": 503, "message": "No provider"}})
    network = Network(unavailable, unavailable, unavailable)
    with pytest.raises(ServiceUnavailableError, match="after 3 attempt"):
        await provider(network, max_retries=2).complete(system="s", user="u")
    assert len(network.requests) == 3


async def test_a_dropped_connection_is_retried() -> None:
    network = Network(
        httpx.ConnectError("connection reset"),
        httpx.Response(200, json=answer()),
    )
    await provider(network).complete(system="s", user="u")
    assert len(network.requests) == 2


def test_retry_after_is_honoured_but_capped() -> None:
    llm = OpenRouterLLM(api_key=KEY, model=MODEL, backoff_seconds=1.0)
    assert llm._delay(1, "3") == 3.0
    assert llm._delay(1, "3600") == llm_module.MAX_RETRY_WAIT_SECONDS
    assert llm._delay(2, None) == 2.0  # exponential: 1, 2, 4 ...


# --- the key ------------------------------------------------------------------


async def test_the_key_never_reaches_an_error_or_a_log(logs: Recorder) -> None:
    network = Network(httpx.Response(401, json={"error": {"code": 401, "message": "bad key"}}))
    with pytest.raises(ServiceUnavailableError) as raised:
        await provider(network).complete(system="s", user="u")

    assert KEY not in str(raised.value)
    assert KEY not in repr(raised.value.__dict__)
    assert all(KEY not in repr(fields) for _, fields in logs.events)


async def test_a_moderation_payload_is_not_logged(logs: Recorder) -> None:
    """`error.metadata` can echo the flagged input, which is our prompt."""
    body = {
        "error": {
            "code": 403,
            "message": "Input flagged",
            "metadata": {"flagged_input": "PATIENT SYN-0001 note text"},
        }
    }
    with pytest.raises(ServiceUnavailableError):
        await provider(Network(httpx.Response(403, json=body))).complete(system="s", user="u")
    assert all("SYN-0001" not in repr(fields) for _, fields in logs.events)


def test_the_key_does_not_appear_when_settings_are_printed() -> None:
    settings = Settings(
        app_env=AppEnv.CI,
        llm_provider=LLMProviderName.OPENROUTER,
        llm_model=MODEL,
        openrouter_api_key=KEY,
    )
    assert KEY not in repr(settings)
    assert KEY not in str(settings.model_dump())


# --- configuration ------------------------------------------------------------


def test_openrouter_without_a_key_refuses_to_start() -> None:
    # Explicitly None: Docker passes .env into the container's environment, so
    # once a real key is there, leaving this out would quietly supply it and
    # the test would stop testing anything.
    with pytest.raises(ValidationError, match="OPENROUTER_API_KEY"):
        Settings(
            app_env=AppEnv.CI,
            llm_provider=LLMProviderName.OPENROUTER,
            llm_model=MODEL,
            openrouter_api_key=None,
        )


def test_an_empty_key_counts_as_no_key() -> None:
    """`.env.example` lists the key with an empty value; that must mean "not set"."""
    with pytest.raises(ValidationError, match="OPENROUTER_API_KEY"):
        Settings(
            app_env=AppEnv.CI,
            llm_provider=LLMProviderName.OPENROUTER,
            llm_model=MODEL,
            openrouter_api_key="",
        )


def test_an_ollama_model_name_is_refused_for_openrouter() -> None:
    """Otherwise the Ollama default would be sent to OpenRouter, and fail on every call."""
    with pytest.raises(ValidationError, match="company/model"):
        Settings(
            app_env=AppEnv.CI,
            llm_provider=LLMProviderName.OPENROUTER,
            llm_model="phi3:latest",
            openrouter_api_key=KEY,
        )


def test_the_factory_builds_the_openrouter_provider() -> None:
    settings = Settings(
        app_env=AppEnv.CI,
        llm_provider=LLMProviderName.OPENROUTER,
        llm_model=MODEL,
        openrouter_api_key=KEY,
    )
    built = build_llm_provider(settings)
    assert isinstance(built, OpenRouterLLM)
    assert built.model == MODEL


def test_other_providers_do_not_need_a_key() -> None:
    Settings(app_env=AppEnv.CI, llm_provider=LLMProviderName.ECHO)
    Settings(app_env=AppEnv.CI, llm_provider=LLMProviderName.OLLAMA, llm_model="phi3:latest")
