"""Instrumentation wrappers.

`MeasuredLLM` is a decorator rather than timing code inside each provider, and
rather than timing at the call sites. There are four call sites - grounded
chat, the agent's planner, the agent's synthesis step and the evaluation
judge - and a provider swapped in later would arrive uninstrumented if the
timing lived in `OllamaLLM`. Wrapping the protocol is the one placement that
covers every caller and every implementation, which is the same argument that
puts the injection filter at the retriever's exit and tool authorisation inside
the registry's dispatch.

What it records: duration, model, and the token counts the provider reported.
Never the prompt or the completion - project plan section 15.
"""

from __future__ import annotations

import time

from app.observability.metrics import METRICS
from app.observability.trace import span
from app.providers.llm import Completion, LLMProvider


class MeasuredLLM:
    """Times every completion and counts it, whatever the provider is."""

    def __init__(self, inner: LLMProvider) -> None:
        self._inner = inner

    @property
    def model(self) -> str:
        return self._inner.model

    async def aclose(self) -> None:
        """Release the provider's connection, if it holds one.

        Only the hosted provider keeps a client open; the others have nothing
        to close, so this asks rather than requiring it of the protocol.
        """
        close = getattr(self._inner, "aclose", None)
        if close is not None:
            await close()

    async def complete(self, *, system: str, user: str) -> Completion:
        started = time.perf_counter()
        with span("llm.complete", model=self._inner.model) as record:
            # Prompt size is the one thing about the prompt worth recording:
            # it is what the context window and the cost are spent on, and it
            # is a number rather than content.
            record.attributes["prompt_chars"] = len(system) + len(user)
            completion = await self._inner.complete(system=system, user=user)
            record.attributes["prompt_tokens"] = completion.prompt_tokens
            record.attributes["completion_tokens"] = completion.completion_tokens

        METRICS.record_llm(
            duration_ms=(time.perf_counter() - started) * 1000,
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
        )
        return completion
