"""Deterministic token counting for the SDK.

This is a verbatim copy of the server's ``app.utils.tokens.estimate_tokens``.
The two implementations must stay in lockstep: if a client reported a different
number than the server would have derived, the cost and efficiency analytics
would silently disagree with the raw text, which is the one thing the project's
anti-fabrication rule forbids.

**No LLM is ever asked to count tokens.** Token counts are deterministic math,
and a model call would be both slower and less accurate than the regex below.
If the host application has exact counts (Ollama returns ``prompt_eval_count``
and ``eval_count``), pass them to ``log_generation`` and they win; this function
is only the fallback.
"""

from __future__ import annotations

import math
import re
from typing import Final

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

# Average characters per token for English prose in BPE vocabularies, and the
# longest regex hit still assumed to be a single token.
_CHARS_PER_TOKEN: Final = 4.0
_MAX_RUN_CHARS: Final = 16


def estimate_tokens(text: str | None) -> int:
    """Approximate the token count of ``text`` without loading any model.

    Deterministic and dependency-free. Kept byte-identical in behaviour to the
    server's ``app.utils.tokens.estimate_tokens`` so that a client-reported
    count and a server-derived count of the same text always agree.
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


__all__ = ["estimate_tokens"]
