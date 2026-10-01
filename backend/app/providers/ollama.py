"""Ollama adapter — the one LLM provider RAGOps requires.

Ollama is local, free and account-free, which is what lets the project run with
no paid API anywhere. Two implementation choices carry the project's rules:

* **``/api/chat``, never ``/api/generate``.** The chat endpoint reports the
  *exact* prompt and completion token counts the model actually consumed, as
  ``prompt_eval_count`` and ``eval_count``. Those numbers go straight onto the
  ``llm_calls`` row. They are only ever recomputed with
  :func:`app.utils.tokens.count_message_tokens` — and flagged inexact — if the
  engine omitted them, which older builds can do.
* **Client-side monotonic timing.** ``latency_ms`` and
  ``time_to_first_token_ms`` come from :func:`time.perf_counter` around the
  awaited call, so a number RAGOps reports is a number RAGOps measured, not one
  a remote process claims.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Final

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from app.config import settings
from app.core.logging import get_logger
from app.providers.base import (
    ChatMessage,
    LLMProvider,
    LLMProviderError,
    LLMResponse,
)
from app.utils.tokens import count_message_tokens

logger = get_logger("ragops.providers.ollama")

# The probe in ``available()`` is a liveness check, not a request; two seconds
# is long enough for a warm local daemon and short enough that a health check
# does not hang on a stopped one.
_AVAILABILITY_TIMEOUT_SECONDS: Final = 2.0

# Everything worth retrying is a transport hiccup or a 5xx/429 from the daemon.
# A 400 means the request itself is wrong and repeating it verbatim would just
# burn the timeout budget three times over.
_RETRYABLE_STATUS: Final = frozenset({408, 425, 429, 500, 502, 503, 504})
_RETRYABLE_EXCEPTIONS: Final = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.RemoteProtocolError,
    httpx.PoolTimeout,
)


def _is_retryable(exc: BaseException) -> bool:
    """Transient transport failure, or a status code worth a second attempt?"""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRYABLE_STATUS
    return isinstance(exc, _RETRYABLE_EXCEPTIONS)


# Two retries with exponential backoff. A local model loading weights can stall
# a socket for a while, and that is exactly the case worth surviving; three
# failures means the daemon is down and the caller should hear about it.
_retry_transient = retry(
    retry=retry_if_exception(_is_retryable),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=8.0),
    reraise=True,
)

# Ollama options that map straight onto the request body. Anything else is a
# caller mistake and is reported as one rather than silently dropped.
_PASSTHROUGH_OPTIONS: Final = (
    "format",
    "stop",
    "keep_alive",
    "seed",
    "top_p",
    "top_k",
    "repeat_penalty",
)


@dataclass(slots=True)
class _ChatResult:
    """Internal carrier: the folded text plus the engine payload behind it."""

    message: str
    raw: dict[str, Any] = field(default_factory=dict)
    time_to_first_token_ms: float | None = None


class OllamaProvider(LLMProvider):
    """Chat completions against a local Ollama daemon over httpx.

    The :class:`httpx.AsyncClient` is created lazily and reused, because a
    client per call throws away connection pooling and TCP handshakes that
    matter for a localhost daemon under load. Tests inject their own client via
    ``http_client``; that client is then left open by :meth:`aclose`, because
    the provider does not own it.
    """

    name = "ollama"

    def __init__(
        self,
        *,
        http_client: httpx.AsyncClient | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout_seconds: float | None = None,
        num_ctx: int | None = None,
        temperature: float | None = None,
    ) -> None:
        self._base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self._model = model or settings.ollama_model
        self._timeout = (
            settings.ollama_timeout_seconds if timeout_seconds is None else timeout_seconds
        )
        self._num_ctx = settings.ollama_num_ctx if num_ctx is None else num_ctx
        self._temperature = (
            settings.ollama_temperature if temperature is None else temperature
        )
        self._client = http_client
        self._owns_client = http_client is None

    # ------------------------------------------------------------------
    # Transport
    # ------------------------------------------------------------------

    @property
    def client(self) -> httpx.AsyncClient:
        """The shared HTTP client, created on first use.

        Built on first access rather than in ``__init__`` so that constructing
        a provider never opens a socket — which is what keeps this module
        importable and instantiable in a test process with no daemon running.
        """
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(self._timeout, connect=10.0),
            )
            self._owns_client = True
        return self._client

    async def aclose(self) -> None:
        """Close the client, but only if this provider created it."""
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------
    # LLMProvider
    # ------------------------------------------------------------------

    async def available(self) -> bool:
        """Cheap ``GET /api/tags`` liveness probe.

        Never raises. An unreachable or slow daemon is a normal state for a
        local-first tool — the deterministic evaluation path has to keep working
        with Ollama stopped — so the answer is ``False`` plus a debug line, not
        an exception that a caller has to remember to catch.
        """
        try:
            response = await self.client.get(
                "/api/tags", timeout=_AVAILABILITY_TIMEOUT_SECONDS
            )
        except Exception as exc:  # noqa: BLE001 - any failure means "not up"
            logger.debug(
                "ollama.unavailable",
                base_url=self._base_url,
                error=f"{type(exc).__name__}: {exc}",
            )
            return False
        if response.status_code >= 400:
            logger.debug(
                "ollama.unavailable",
                base_url=self._base_url,
                status=response.status_code,
            )
            return False
        return True

    async def list_models(self) -> list[str]:
        """Model tags the daemon has pulled, sorted for stable display.

        Returns ``[]`` rather than raising: an unavailable daemon simply has no
        models to list, and callers render that as "no models installed" rather
        than as an error.
        """
        try:
            response = await self.client.get(
                "/api/tags", timeout=_AVAILABILITY_TIMEOUT_SECONDS
            )
        except Exception as exc:  # noqa: BLE001 - discovery is best-effort
            logger.debug(
                "ollama.list_models_failed",
                base_url=self._base_url,
                error=f"{type(exc).__name__}: {exc}",
            )
            return []
        if response.status_code >= 400:
            return []
        try:
            payload = response.json()
        except ValueError:
            logger.debug("ollama.list_models_malformed", base_url=self._base_url)
            return []
        if not isinstance(payload, dict):
            return []
        models = payload.get("models")
        if not isinstance(models, list):
            return []
        names = {
            str(entry["name"])
            for entry in models
            if isinstance(entry, dict) and entry.get("name")
        }
        return sorted(names)

    async def generate(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
        **opts: Any,
    ) -> LLMResponse:
        """One chat completion, carrying the engine's exact token counts.

        ``opts`` accepted beyond the base signature:

        ``stream``
            Accumulate the NDJSON stream and return the same
            :class:`LLMResponse` shape as the non-streaming path, additionally
            populating ``time_to_first_token_ms``. Defaults to ``False`` —
            streaming is only worth its overhead when a caller wants the
            first-token number, which the RAG demo path and the judge do.
        ``format``
            Ollama structured-output mode, e.g. ``"json"``.
        ``stop``
            Sequence list that ends generation.
        ``keep_alive``
            How long the daemon keeps the model resident. Left unset by default
            so RAGOps never silently changes the user's model lifetime.
        """
        if not messages:
            raise LLMProviderError("generate() requires at least one message")

        resolved_model = model or self._model
        resolved_temperature = (
            self._temperature if temperature is None else float(temperature)
        )
        stream = bool(opts.pop("stream", False))

        options: dict[str, Any] = {
            "temperature": resolved_temperature,
            "num_ctx": self._num_ctx,
        }
        if max_tokens is not None:
            # Ollama's name for an output-token cap. Sent only when the caller
            # asked for one, so the model's own default is otherwise untouched.
            options["num_predict"] = int(max_tokens)

        payload: dict[str, Any] = {
            "model": resolved_model,
            "messages": list(messages),
            "stream": stream,
            "options": options,
        }
        for key in _PASSTHROUGH_OPTIONS:
            value = opts.pop(key, None)
            if value is not None:
                payload[key] = value
        if opts:
            # Surfacing unknown knobs beats silently dropping them: a typo in a
            # caller's options would otherwise look like the model ignored it.
            raise LLMProviderError(f"unsupported Ollama options: {sorted(opts)}")

        started = time.perf_counter()
        if stream:
            result = await self._post_stream("/api/chat", payload)
        else:
            result = await self._post_json("/api/chat", payload)
        latency_ms = (time.perf_counter() - started) * 1000.0

        return self._to_response(
            result,
            model=resolved_model,
            latency_ms=latency_ms,
            messages=messages,
        )

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------

    @_retry_transient
    async def _post_json(self, path: str, payload: dict[str, Any]) -> _ChatResult:
        """POST and return a parsed non-streaming chat result."""
        response = await self.client.post(path, json=payload)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = _error_detail(response)
            logger.warning(
                "ollama.request_failed",
                model=payload["model"],
                status=exc.response.status_code,
                error=detail,
            )
            raise LLMProviderError(
                f"Ollama returned {exc.response.status_code}: {detail}"
            ) from exc
        try:
            body = response.json()
        except ValueError as exc:
            raise LLMProviderError("Ollama returned a non-JSON response") from exc
        if not isinstance(body, dict):
            raise LLMProviderError("Ollama returned a JSON value that is not an object")
        return _ChatResult(message=_extract_message(body), raw=body)

    @_retry_transient
    async def _post_stream(self, path: str, payload: dict[str, Any]) -> _ChatResult:
        """POST and fold the NDJSON stream into one result.

        Ollama emits one JSON object per line. Only the objects that carry
        message content contribute text; the final object carries the token
        counts. Time to first token is stamped at the first chunk with
        content, which is the definition of the metric — the empty preamble
        line Ollama sends while the model loads is not a token.
        """
        chunks: list[str] = []
        final: dict[str, Any] = {}
        ttft_ms: float | None = None
        started = time.perf_counter()

        async with self.client.stream("POST", path, json=payload) as response:
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                detail = _error_detail(await response.aread().as_response(response))
                logger.warning(
                    "ollama.request_failed",
                    model=payload["model"],
                    status=exc.response.status_code,
                    error=detail,
                    stream=True,
                )
                raise LLMProviderError(
                    f"Ollama returned {exc.response.status_code}: {detail}"
                ) from exc
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                try:
                    chunk = json.loads(line)
                except ValueError:
                    # A truncated line is the signature of a connection dropped
                    # mid-stream, which is retryable; the retry predicate cannot
                    # see a parse failure, so it is re-raised as one.
                    raise httpx.RemoteProtocolError(
                        f"unparseable stream line: {line[:120]}"
                    ) from None
                if not isinstance(chunk, dict):
                    raise LLMProviderError("Ollama stream chunk is not an object")
                piece = _extract_message(chunk)
                if piece:
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - started) * 1000.0
                    chunks.append(piece)
                if chunk.get("done"):
                    final = chunk

        text = "".join(chunks)
        raw = final or {"model": payload["model"], "message": {"content": text}}
        return _ChatResult(
            message=text, raw=raw, time_to_first_token_ms=ttft_ms
        )

    # ------------------------------------------------------------------
    # Response mapping
    # ------------------------------------------------------------------

    def _to_response(
        self,
        result: _ChatResult,
        *,
        model: str,
        latency_ms: float,
        messages: list[ChatMessage],
    ) -> LLMResponse:
        """Build an :class:`LLMResponse` from the engine payload.

        The engine's counts are authoritative. The fallback to
        :func:`count_message_tokens` fires only when ``prompt_eval_count`` is
        absent, and it sets ``token_counts_exact=False`` so a persisted row can
        never be mistaken for a measured figure.
        """
        raw = result.raw
        input_tokens = _int_or_none(raw.get("prompt_eval_count"))
        output_tokens = _int_or_none(raw.get("eval_count"))

        exact = input_tokens is not None
        if not exact:
            input_tokens = count_message_tokens(messages)
        if output_tokens is None:
            output_tokens = 0
            exact = False

        resolved_model = str(raw.get("model") or model)
        ttft = result.time_to_first_token_ms
        logger.debug(
            "ollama.generate",
            model=resolved_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=round(latency_ms, 2),
            time_to_first_token_ms=None if ttft is None else round(ttft, 2),
            token_counts_exact=exact,
        )

        return LLMResponse(
            text=result.message,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            latency_ms=round(latency_ms, 3),
            model=resolved_model,
            provider=self.name,
            time_to_first_token_ms=None if ttft is None else round(ttft, 3),
            raw=raw,
            token_counts_exact=exact,
        )


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------


def _extract_message(payload: dict[str, Any]) -> str:
    """Pull the assistant text out of an Ollama chat object, defensively."""
    message = payload.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
    return ""


def _int_or_none(value: Any) -> int | None:
    """Coerce a JSON number to ``int``, or ``None`` if absent, negative or garbage."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value < 0:
        return None
    return int(value)


def _error_detail(response: httpx.Response) -> str:
    """Best-effort extraction of Ollama's error message from a response body."""
    try:
        body = response.json()
    except ValueError:
        return (response.text or "")[:200]
    if isinstance(body, dict):
        return str(body.get("error") or body)[:200]
    return str(body)[:200]
