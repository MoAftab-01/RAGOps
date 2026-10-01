"""Deterministic token accounting.

Token counts are never produced by an LLM. They come from one of two
deterministic paths, in priority order:

1. **Real tokenizer** — if a HuggingFace tokenizer for the model is already
   present in the local cache (``transformers.AutoTokenizer``), use it. Ollama
   itself also reports exact counts, which is preferred when available.
2. **Heuristic counter** — the default. A documented, dependency-free
   approximation that needs no model download and behaves identically on the
   SDK and the server.

The heuristic over-counts CJK and under-counts rare subwords; see
:func:`estimate_tokens` for the stated accuracy. It is a *consistent* unit,
which is what analytics needs: ratios, percentiles and trend lines stay valid
even when absolute values are off by a few percent.
"""

from __future__ import annotations

import math
import re
import threading
from functools import lru_cache
from typing import Final

# Words, numbers, and runs of punctuation/whitespace each count as a token.
# This tracks BPE tokenisers on English prose to within a few percent.
_TOKEN_RE: Final = re.compile(
    r"""
    \d+                       # numbers (often split by BPE, counted as one here)
    | [A-Za-z]+               # words
    | [^\sA-Za-z\d]           # single punctuation / symbol
    """,
    re.VERBOSE,
)

# Whitespace is a *separator*, not a token. The separator regex is tried
# first so a space between two words is consumed once instead of being
# counted by both the word branch and a whitespace branch.
_WS_RE: Final = re.compile(r"\s+")

# CJK ideographs, kana and Hangul are roughly one token per character for
# BPE vocabularies built for them, which word-splitting badly under-counts.
_CJK_RE: Final = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")

# Average characters per token for English prose in BPE vocabularies.
# Used only for the cheap "how big is this prompt" guard on very long inputs.
_CHARS_PER_TOKEN: Final = 4.0

# Longest regex hit still assumed to be a single token. Real BPE vocabularies
# top out around 16 characters per token, so anything past this is subdivided.
_MAX_RUN_CHARS: Final = 16

_tokenizer_cache: dict[str, object] = {}
_tokenizer_lock = threading.Lock()


def estimate_tokens(text: str | None) -> int:
    """Approximate the token count of ``text`` without loading any model.

    Deterministic and dependency-free. This is the function the SDK and the
    collector both call, so client-reported and server-computed counts agree.
    """
    if not text:
        return 0

    cjk = len(_CJK_RE.findall(text))
    # Remove CJK so the word regex does not also match those characters, then
    # collapse every whitespace run to a single space so the separator is
    # consumed exactly once regardless of how the text was indented.
    remainder = _WS_RE.sub(" ", _CJK_RE.sub(" ", text))

    matches = list(_TOKEN_RE.finditer(remainder))
    tokens = len(matches)

    # A single regex hit can be an unbounded run -- "aaaa...a" is one
    # [A-Za-z]+ match, but no BPE vocabulary encodes 400 characters as one
    # token. Subdivide any hit far wider than a plausible token so long words
    # and blob-like input (base64, minified JS, hashes) cannot collapse to a
    # single token and silently undercount the whole request.
    for match in matches:
        span = len(match.group())
        if span > _MAX_RUN_CHARS:
            tokens += math.ceil(span / _CHARS_PER_TOKEN) - 1

    return tokens + cjk


@lru_cache(maxsize=4)
def _load_tokenizer(model_name: str) -> object | None:
    """Load a cached HF tokenizer, or ``None`` if it is not available offline.

    Never downloads: a missing model must not turn a telemetry call into a
    network call, so an offline cache miss simply falls back to the heuristic.
    """
    with _tokenizer_lock:
        if model_name in _tokenizer_cache:
            return _tokenizer_cache[model_name]

        tokenizer: object | None = None
        try:
            from transformers import AutoTokenizer  # noqa: PLC0415

            candidate = AutoTokenizer.from_pretrained(
                model_name, local_files_only=True
            )
            tokenizer = candidate
        except Exception:
            # Not cached, transformers missing, or the repo is gated.
            tokenizer = None

        _tokenizer_cache[model_name] = tokenizer
        return tokenizer


def count_tokens(text: str | None, model_name: str | None = None) -> int:
    """Token count for ``text``, using a real tokenizer when one is cached.

    ``model_name`` should be an Ollama tag (``qwen2.5:3b``) or a HF repo id. The
    suffix after ``:`` is stripped because ``qwen2.5:3b`` is not a HF repo.
    """
    if not text:
        return 0
    if model_name:
        repo = model_name.split(":", 1)[0]
        tokenizer = _load_tokenizer(repo)
        if tokenizer is not None:
            try:
                return int(len(tokenizer.encode(text, add_special_tokens=False)))
            except Exception:
                pass
    return estimate_tokens(text)


def count_message_tokens(messages: list[dict[str, str]]) -> int:
    """Token count for a chat-message list, including per-message overhead.

    Chat formats wrap every turn in role/separator tokens; 4 per message is the
    commonly used constant for the formats in play here.
    """
    if not messages:
        return 0
    per_message_overhead = 4
    total = per_message_overhead * len(messages)
    total += 3  # assistant priming / reply-begin tokens
    for message in messages:
        total += count_tokens(message.get("content"))
    return total


def truncate_to_tokens(
    text: str, max_tokens: int, model_name: str | None = None
) -> str:
    """Truncate ``text`` to at most ``max_tokens`` tokens.

    Character-based, so it can overshoot slightly on CJK; good enough for
    context-window guards where the exact boundary is not critical.
    """
    if max_tokens <= 0 or not text:
        return ""
    approx = count_tokens(text, model_name)
    if approx <= max_tokens:
        return text
    # Scale by characters/token, then trim on a word boundary where possible.
    limit = int(max_tokens * _CHARS_PER_TOKEN)
    truncated = text[:limit]
    if " " in truncated:
        truncated = truncated[: truncated.rfind(" ")]
    return truncated


def token_efficiency(
    *,
    input_tokens: int,
    context_tokens: int,
    output_tokens: int,
    retrieved_documents: list[str] | None = None,
    duplicate_document_ids: set[str] | None = None,
) -> dict[str, float | int | list[str]]:
    """Score how much of a request's prompt budget is doing useful work.

    Returns a 0-100 score plus the measured components. The score is a weighted
    reduction of four independently computed ratios, not an estimate of money
    saved — the *potential waste* percentage is the fraction of input tokens
    that are either duplicated across retrieved documents or supplied as
    context the generation then did not need to read.

    ``output_tokens / input_tokens`` is included because a very low ratio means
    the request paid for a large prompt and returned almost nothing, which is
    the signature of a context that was too large to be read.
    """
    retrieved_documents = retrieved_documents or []
    duplicate_document_ids = duplicate_document_ids or set()

    total_input = max(0, input_tokens)
    total_output = max(0, output_tokens)
    context = max(0, context_tokens)

    # 1. Duplicate content: measured on document ids, not guessed.
    duplicate_ratio = (
        len(duplicate_document_ids) / len(retrieved_documents)
        if retrieved_documents
        else 0.0
    )

    # 2. Context share: how much of the prompt is retrieved context.
    context_share = context / total_input if total_input else 0.0

    # 3. Output yield: tokens produced per input token spent. 0.25 in/1 out is a
    #    healthy RAG summary; below 0.05 the prompt is mostly being ignored.
    output_yield = total_output / total_input if total_input else 0.0
    yield_component = min(1.0, output_yield / 0.25) if total_input else 1.0

    # 4. Context headroom: a prompt that is mostly context is over-fetching.
    #    Full marks below 60% context share, linear decay to zero at 100%.
    headroom_component = max(0.0, min(1.0, (1.0 - context_share) / 0.4))

    # Weighted sum. Duplicate content is the strongest single signal, so it
    # carries the most weight.
    score = 100.0 * (
        0.40 * (1.0 - duplicate_ratio)
        + 0.25 * yield_component
        + 0.35 * headroom_component
    )
    score = max(0.0, min(100.0, score))

    # Potential waste: duplicated document tokens plus context the model did not
    # turn into output. Both components are measured, so the number is bounded
    # and explainable rather than a projection.
    duplicate_tokens = int(context * duplicate_ratio) if context else 0
    unconverted_context = max(0, context - total_output * 4)  # ~4 output tok / ctx tok
    wasted = min(total_input, duplicate_tokens + unconverted_context)
    potential_waste_pct = 100.0 * wasted / total_input if total_input else 0.0

    return {
        "score": round(score, 1),
        "duplicate_document_count": len(duplicate_document_ids),
        "duplicate_ratio": round(duplicate_ratio, 4),
        "duplicate_tokens": duplicate_tokens,
        "context_share": round(context_share, 4),
        "output_yield": round(output_yield, 4),
        "potential_waste_pct": round(potential_waste_pct, 1),
        "wasted_tokens": wasted,
    }
