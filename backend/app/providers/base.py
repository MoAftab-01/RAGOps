"""The provider seam — one response shape for every LLM backend.

RAGOps needs exactly one LLM provider: a local Ollama on localhost, because the
project must run with no paid API keys and no accounts. This module exists so
that adding a *second* backend later is an addition rather than a rewrite — a
cloud adapter subclasses :class:`LLMProvider`, returns an :class:`LLMResponse`,
and gets one entry in ``app.providers.registry``. No cloud adapter is required,
and none ships here.

Two invariants are enforced by construction rather than by convention:

* **Token counts are never invented.** A provider reports the counts its engine
  actually measured. An implementation may fall back to
  :func:`app.utils.tokens.count_tokens` only when the engine omits them, and
  must then set ``token_counts_exact=False`` so that nothing downstream stores
  a heuristic figure as though it were measured.
* **Latency is measured client-side with a monotonic clock**
  (:func:`time.perf_counter`), so server-side queue time cannot hide a slow
  network hop and the number is unaffected by wall-clock adjustments.

``LLMResponse`` is a plain frozen dataclass rather than a Pydantic model on
purpose: providers are called on the hot path of the collector, and the shape
is written straight onto an ``llm_calls`` row.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, ClassVar

# One chat turn. Kept deliberately loose — Ollama accepts extra keys per message
# (for example ``images``) and a TypedDict would force callers to strip them.
ChatMessage = dict[str, str]


class LLMProviderError(RuntimeError):
    """A generation call failed for a reason the caller can act on.

    Wraps transport and protocol failures so that callers — notably the opt-in
    ``LLMJudge``, which must return ``None`` rather than raise into the
    deterministic evaluation path — can catch a single type.
    """


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """One generation, fully accounted for.

    ``input_tokens``/``output_tokens``/``total_tokens`` are exact engine counts
    whenever the provider supplies them, which is the whole reason RAGOps talks
    to ``/api/chat`` rather than to a completion endpoint. ``raw`` keeps the
    untouched provider payload so callers can read fields RAGOps does not model
    (``done_reason``, per-phase durations) without re-issuing the request.
    """

    text: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_ms: float
    model: str
    provider: str
    time_to_first_token_ms: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    # Strictly additive: tells a caller whether the token figures came from the
    # engine or from the documented heuristic fallback, so a persisted count is
    # never mistaken for a measured one.
    token_counts_exact: bool = True


class LLMProvider(abc.ABC):
    """Abstract adapter over a chat-completion backend.

    This is the seam a cloud provider slots into. An adapter has three
    obligations and nothing else: turn messages into a completion, report
    whether the backend is reachable, and enumerate what it can serve. All
    RAGOps-specific behaviour (which model, which temperature, whether the
    judge is allowed to run at all) belongs to the caller, not the adapter —
    an adapter that "helpfully" enforces policy is no longer swappable.
    """

    # Registry key. Subclasses must set it; ``registry.get_provider`` uses it.
    name: ClassVar[str] = ""

    @abc.abstractmethod
    async def generate(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
        **opts: Any,
    ) -> LLMResponse:
        """Run one chat completion.

        ``None`` for an optional argument means "use the configured default",
        never "omit the parameter" — an adapter resolves the default from
        ``app.config.settings``. Backend-specific extras ride in ``opts`` (the
        Ollama adapter reads ``stream``, ``format``, ``stop`` and
        ``keep_alive`` from it) so that adding a knob to one backend does not
        widen this signature for all of them.

        Raises :class:`LLMProviderError` on failure. Providers never return an
        empty response to signal an error.
        """

    @abc.abstractmethod
    async def available(self) -> bool:
        """Whether the backend is reachable right now.

        Must be a cheap, bounded probe and must return ``False`` rather than
        raise: callers use it to decide whether an *optional* signal is
        available, and an exception would be a worse answer than ``False``.
        """

    @abc.abstractmethod
    async def list_models(self) -> list[str]:
        """Model tags this backend currently serves.

        Empty list when the backend cannot be reached — this is discovery
        metadata, and a failure to list is not a failure of the application.
        """

    async def aclose(self) -> None:
        """Release any transport resources the adapter owns.

        Not abstract: most adapters are stateless. The default is a no-op so
        that callers can unconditionally await cleanup on any provider.
        """
        return None
