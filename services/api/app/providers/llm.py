"""Generation.

Three providers behind one protocol:

* `OpenRouterLLM` - hosted models through OpenRouter. The only provider whose
  prompts leave this machine (decision 48).
* `OllamaLLM` - a local model through Ollama.
* `EchoLLM` - a deterministic stub that composes an answer directly from the
  retrieved context, with correct citations and no model at all.

The stub is not a mock hidden in the test suite - it is a configured provider,
because it makes the *whole* pipeline runnable in CI and on a machine without
5GB to spare for model weights. Everything except the generated prose is
genuinely exercised: retrieval, filtering, fusion, reranking, citation parsing,
persistence, cost accounting. When Sprint 5 measures retrieval metrics, the
stub is also the right choice - it removes the model as a source of variance
between runs.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from app.core.config import LLMProvider as LLMProviderName
from app.core.config import Settings
from app.core.errors import ServiceUnavailableError
from app.core.logging import get_logger

logger = get_logger(__name__)

# Local models are free to run; the field exists so the same accounting works
# when an Azure OpenAI deployment is swapped in.
LOCAL_COST_PER_1K_TOKENS = 0.0


@dataclass(frozen=True, slots=True)
class Completion:
    text: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    estimated_cost_usd: float | None = None


class LLMProvider(Protocol):
    @property
    def model(self) -> str: ...

    async def complete(self, *, system: str, user: str) -> Completion: ...


class EchoLLM:
    """Builds an answer from the context without a model.

    Quotes the first sentence of each supplied block and cites it. The output is
    not fluent, but it is grounded and correctly cited by construction, which is
    exactly what the pipeline around it needs to be tested against.
    """

    @property
    def model(self) -> str:
        return "echo"

    async def complete(self, *, system: str, user: str) -> Completion:
        blocks = _parse_context_blocks(user)
        if not blocks:
            text = (
                "No supporting documents were found within your access scope, "
                "so I cannot answer this question."
            )
        else:
            lines = [
                "Based on the retrieved documents:",
                *(f"- {summary} [{number}]" for number, summary in blocks),
            ]
            text = "\n".join(lines)

        return Completion(
            text=text,
            model="echo",
            prompt_tokens=_estimate_tokens(system) + _estimate_tokens(user),
            completion_tokens=_estimate_tokens(text),
            estimated_cost_usd=0.0,
        )


class OllamaLLM:
    """Local generation through Ollama's /api/chat endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout_seconds: float = 120.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout_seconds

    @property
    def model(self) -> str:
        return self._model

    async def complete(self, *, system: str, user: str) -> Completion:
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {
                # Temperature 0: an evaluation run has to be reproducible, and
                # a grounded answer should not be creative.
                "temperature": self._temperature,
                "num_predict": self._max_tokens,
            },
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(f"{self._base_url}/api/chat", json=payload)
                response.raise_for_status()
                body = response.json()
        except httpx.HTTPStatusError as exc:
            detail = (
                f"Ollama has no model '{self._model}'. Pull it with: ollama pull {self._model}"
                if exc.response.status_code == 404
                else f"Ollama returned {exc.response.status_code}."
            )
            raise ServiceUnavailableError(detail) from exc
        except httpx.HTTPError as exc:
            raise ServiceUnavailableError(
                f"Could not reach Ollama at {self._base_url}. "
                f"Is it running, and is OLLAMA_BASE_URL correct?"
            ) from exc

        return Completion(
            text=str(body.get("message", {}).get("content", "")).strip(),
            model=self._model,
            prompt_tokens=body.get("prompt_eval_count"),
            completion_tokens=body.get("eval_count"),
            estimated_cost_usd=LOCAL_COST_PER_1K_TOKENS,
        )


# The request was fine and the moment was not: worth another attempt. Every
# other 4xx means the request itself is wrong - a bad key, no credits, a
# refused prompt - and repeating it only repeats the refusal.
RETRYABLE_STATUSES = frozenset({408, 429, 502, 503})
# However long a Retry-After header asks for, a user is waiting on the other end.
MAX_RETRY_WAIT_SECONDS = 20.0
# Provider error messages are logged, cut to this length. They carry the reason
# for a refusal; nothing from our side of the conversation should be in them,
# and bounding them keeps it that way even if a provider disagrees.
MAX_LOGGED_MESSAGE = 200


class OpenRouterLLM:
    """Hosted models through OpenRouter's OpenAI-compatible chat API.

    **Prompts leave this machine.** The question and the retrieved document text
    go to OpenRouter and on to the model's provider. The corpus is synthetic by
    rule, so this is a choice about the project's local-only property, not about
    patient data (decision 48). Embeddings and reranking stay local.

    **The key is only ever a header.** It is set once on the client, never put
    in a log line or an error message, and comes from settings as a `SecretStr`.

    **An error can arrive with HTTP 200.** OpenRouter documents that a provider
    failing after the response has started gives a `200 OK` whose body holds an
    `error` object and no `choices`. Trusting the status code would turn that
    into an empty answer that looks like a model with nothing to say, so the
    body is checked before it is read.

    **Cost is what OpenRouter says it was.** `usage.cost` is returned with every
    non-streaming completion. When it is missing, the cost is recorded as
    unknown (`None`) rather than as zero, because a dashboard that sums zeros
    for calls it could not price is reporting a number nobody measured.

    One client is kept for the life of the provider, so repeated calls reuse
    the TLS connection instead of paying a handshake each time; `aclose`
    releases it at shutdown.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "https://openrouter.ai/api/v1",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        backoff_seconds: float = 1.0,
        reasoning: str = "off",
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._api_key = api_key
        self._reasoning = reasoning
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._backoff = backoff_seconds
        # Injectable so the tests can answer without a network or a key.
        self._transport = transport
        self._client: httpx.AsyncClient | None = None

    @property
    def model(self) -> str:
        return self._model

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def complete(self, *, system: str, user: str) -> Completion:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            # Temperature 0 for the same reason as the local provider: a
            # grounded answer should not be creative, and an evaluation run
            # should be as repeatable as the model allows.
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }
        reasoning = _reasoning_field(self._reasoning)
        if reasoning is not None:
            payload["reasoning"] = reasoning
        body = await self._post(payload)

        choices = body.get("choices")
        if body.get("error") or not choices:
            raise self._failure(200, body)

        choice = choices[0]
        message = choice.get("message") or {}
        finish_reason = choice.get("finish_reason")
        text = str(message.get("content") or "").strip()
        if finish_reason == "length":
            # A cut-off answer can end mid-citation, and whoever reads the logs
            # should know why.
            logger.warning("llm_truncated", model=self._model, max_tokens=self._max_tokens)

        if not text:
            # Never hand a blank answer upward. OpenRouter documents the cause
            # that was measured here: hidden reasoning used the whole
            # `max_tokens` budget, so `content` is empty and `finish_reason` is
            # "length" - and the reasoning was still billed. A blank answer in
            # the UI looks like a system with nothing to say; an error says what
            # to change.
            logger.warning("llm_empty_answer", model=self._model, finish_reason=finish_reason)
            raise ServiceUnavailableError(
                "The model used its whole output allowance before writing an answer. "
                "Set OPENROUTER_REASONING=off, or raise LLM_MAX_TOKENS."
                if finish_reason == "length"
                else "The model returned an empty answer."
            )

        usage = body.get("usage") or {}
        cost = usage.get("cost")
        return Completion(
            text=text,
            model=self._model,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            estimated_cost_usd=float(cost) if isinstance(cost, int | float) else None,
        )

    # --- transport ----------------------------------------------------------

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout,
                transport=self._transport,
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
        return self._client

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        attempt = 0
        while True:
            try:
                response = await self._http().post("/chat/completions", json=payload)
            except httpx.TransportError as exc:
                # Timeouts and connection failures. Retried, because a dropped
                # connection says nothing about whether the request was valid.
                # The price: a request that timed out may still have been
                # answered and billed, so a retry can cost up to
                # (max_retries + 1) calls. At cents per call, that is the
                # right trade against failing a user's question.
                if attempt < self._max_retries:
                    attempt += 1
                    logger.warning("llm_retry", reason=type(exc).__name__, attempt=attempt)
                    await asyncio.sleep(self._delay(attempt, None))
                    continue
                raise ServiceUnavailableError(
                    f"Could not reach OpenRouter after {attempt + 1} attempt(s)."
                ) from exc

            if response.status_code in RETRYABLE_STATUSES and attempt < self._max_retries:
                attempt += 1
                logger.warning("llm_retry", status=response.status_code, attempt=attempt)
                await asyncio.sleep(self._delay(attempt, response.headers.get("retry-after")))
                continue

            if response.status_code >= 400:
                raise self._failure(response.status_code, _json_or_empty(response), attempt + 1)

            body = _json_or_empty(response)
            if not body:
                raise ServiceUnavailableError("OpenRouter returned a response that is not JSON.")

            # An error inside a 200 is the same failure as the status code
            # would have been, so a transient one gets the same retry.
            in_body = body.get("error")
            in_body_code = in_body.get("code") if isinstance(in_body, dict) else None
            if in_body_code in RETRYABLE_STATUSES and attempt < self._max_retries:
                attempt += 1
                logger.warning("llm_retry", status=200, code=in_body_code, attempt=attempt)
                await asyncio.sleep(self._delay(attempt, None))
                continue
            if in_body is not None:
                raise self._failure(200, body, attempt + 1)
            return body

    def _delay(self, attempt: int, retry_after: str | None) -> float:
        """Honour Retry-After when the server sends one, else back off exponentially."""
        if retry_after is not None:
            try:
                return min(max(float(retry_after), 0.0), MAX_RETRY_WAIT_SECONDS)
            except ValueError:
                pass  # An HTTP date, which OpenRouter does not send; fall through.
        return min(self._backoff * 2 ** (attempt - 1), MAX_RETRY_WAIT_SECONDS)

    def _failure(
        self, status: int, body: dict[str, Any], attempts: int = 1
    ) -> ServiceUnavailableError:
        raw_error = body.get("error")
        error: dict[str, Any] = raw_error if isinstance(raw_error, dict) else {}
        provider_message = str(error.get("message") or "")
        # For an error inside a 200, the code in the body is the real status.
        raw_code = error.get("code")
        code = raw_code if isinstance(raw_code, int) else status

        logger.warning(
            "llm_provider_error",
            provider="openrouter",
            status=status,
            code=code,
            # Only the message. `error.metadata` can carry a moderation
            # provider's view of the flagged input, which is our prompt.
            message=provider_message[:MAX_LOGGED_MESSAGE],
        )
        return ServiceUnavailableError(_describe(code, provider_message, self._model, attempts))


def _describe(code: int, provider_message: str, model: str, attempts: int) -> str:
    """What went wrong, and what to change. Never the key, never the prompt."""
    lowered = provider_message.lower()
    if code == 401:
        return "OpenRouter rejected the API key. Check OPENROUTER_API_KEY in .env."
    if code == 402:
        return "The OpenRouter account has run out of credits."
    if code == 403:
        return "OpenRouter refused the request (a moderation or permission rule)."
    if code == 404 or (code == 400 and "model" in lowered):
        return f"OpenRouter does not recognise the model '{model}'. Check LLM_MODEL."
    if code == 400:
        return "OpenRouter rejected the request as invalid."
    if code in RETRYABLE_STATUSES:
        return (
            f"OpenRouter is rate-limiting or unavailable (HTTP {code}) after {attempts} attempt(s)."
        )
    return f"OpenRouter returned an error (HTTP {code})."


def _reasoning_field(setting: str) -> dict[str, Any] | None:
    """OpenRouter's `reasoning` request field for a setting, or None to omit it.

    "off" uses `enabled: false`, not `effort: "none"`: OpenRouter documents
    both, but each model lists the efforts it supports, and
    deepseek-v4.1-flash lists only max/high/low, so "none" is not one of them.
    """
    if setting == "default":
        return None
    if setting == "off":
        return {"enabled": False}
    return {"effort": setting}


def _json_or_empty(response: httpx.Response) -> dict[str, Any]:
    try:
        body = response.json()
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}


_BLOCK = re.compile(r"^\[(\d{1,2})\] \([^)]*\)\n(.+?)(?=\n\n\[|\n\nQUESTION:|\Z)", re.S | re.M)


def _parse_context_blocks(user_prompt: str) -> list[tuple[int, str]]:
    """Pull (number, first sentence) out of the rendered context blocks."""
    blocks = []
    for match in _BLOCK.finditer(user_prompt):
        body = " ".join(match.group(2).split())
        sentence = body.split(". ")[0].rstrip(".")
        blocks.append((int(match.group(1)), f"{sentence}."))
    return blocks


def _estimate_tokens(text: str) -> int:
    return max(1, int(len(text.split()) / 0.75))


def build_llm_provider(settings: Settings) -> LLMProvider:
    if settings.llm_provider is LLMProviderName.ECHO:
        return EchoLLM()
    if settings.llm_provider is LLMProviderName.OPENROUTER:
        # The settings validator already refuses openrouter without a key; this
        # is the same check where the key is used, not an `assert` that
        # `python -O` would strip.
        if settings.openrouter_api_key is None:
            raise ServiceUnavailableError("LLM_PROVIDER=openrouter needs OPENROUTER_API_KEY.")
        return OpenRouterLLM(
            api_key=settings.openrouter_api_key.get_secret_value(),
            model=settings.llm_model,
            base_url=str(settings.openrouter_base_url),
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout_seconds=settings.openrouter_timeout_seconds,
            max_retries=settings.openrouter_max_retries,
            reasoning=settings.openrouter_reasoning,
        )
    return OllamaLLM(
        base_url=str(settings.ollama_base_url),
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )
