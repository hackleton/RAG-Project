"""Generation.

`OllamaLLM` is the real local provider. `EchoLLM` is a deterministic stub that
composes an answer directly from the retrieved context, with correct citations
and no model at all.

The stub is not a mock hidden in the test suite - it is a configured provider,
because it makes the *whole* pipeline runnable in CI and on a machine without
5GB to spare for model weights. Everything except the generated prose is
genuinely exercised: retrieval, filtering, fusion, reranking, citation parsing,
persistence, cost accounting. When Sprint 5 measures retrieval metrics, the
stub is also the right choice - it removes the model as a source of variance
between runs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

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
    return OllamaLLM(
        base_url=str(settings.ollama_base_url),
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )
