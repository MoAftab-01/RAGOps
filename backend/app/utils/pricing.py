"""Simulated model pricing.

Local Ollama inference is free, so there is no real spend to report. RAGOps
still computes a dollar figure so teams can answer "what would this have cost
on a metered provider, and is the bigger model worth it?" — a question that is
the entire point of the cost/quality page.

The prices below are a bundled default table, clearly a *simulation*. Every
surface that displays a dollar amount labels it as estimated. Users override
the table in the ``models`` table, so the numbers are never treated as truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Final

from app.config import settings


@dataclass(frozen=True, slots=True)
class ModelPricing:
    """Per-1M-token list price for one model, in a given currency."""

    model_name: str
    provider: str
    input_cost_per_1k: float
    output_cost_per_1k: float
    context_window: int | None = None
    is_local: bool = True
    description: str = ""


# Reference list prices, USD per 1K tokens, used only for simulation. Chosen to
# mirror widely published public price lists so the comparisons are directionally
# meaningful; they are NOT a quote and are not fetched from any provider API.
DEFAULT_PRICING: Final[tuple[ModelPricing, ...]] = (
    ModelPricing(
        model_name="qwen2.5:0.5b",
        provider="ollama",
        input_cost_per_1k=0.0,
        output_cost_per_1k=0.0,
        context_window=32768,
        description="Smallest local model. Fastest, weakest retrieval grounding.",
    ),
    ModelPricing(
        model_name="qwen2.5:3b",
        provider="ollama",
        input_cost_per_1k=0.0,
        output_cost_per_1k=0.0,
        context_window=32768,
        description="Default local model. Good quality/latency balance on CPU or laptop GPU.",
    ),
    ModelPricing(
        model_name="qwen2.5:7b",
        provider="ollama",
        input_cost_per_1k=0.0,
        output_cost_per_1k=0.0,
        context_window=32768,
        description="Larger local model. Needs 8GB+ of RAM; noticeably better reasoning.",
    ),
    ModelPricing(
        model_name="llama3:latest",
        provider="ollama",
        input_cost_per_1k=0.0,
        output_cost_per_1k=0.0,
        context_window=8192,
        description="Meta Llama 3 8B via Ollama.",
    ),
    ModelPricing(
        model_name="mistral:latest",
        provider="ollama",
        input_cost_per_1k=0.0,
        output_cost_per_1k=0.0,
        context_window=32768,
        description="Mistral 7B via Ollama.",
    ),
    ModelPricing(
        model_name="gemma2:2b",
        provider="ollama",
        input_cost_per_1k=0.0,
        output_cost_per_1k=0.0,
        context_window=8192,
        description="Google Gemma 2 2B via Ollama.",
    ),
    # --- Simulated metered reference points -------------------------------
    # Non-zero on purpose: they exist so the cost/quality page has a y-axis
    # with structure. Marked is_local=False so the UI labels them as external.
    ModelPricing(
        model_name="gpt-4o-mini",
        provider="openai",
        input_cost_per_1k=0.00015,
        output_cost_per_1k=0.00060,
        context_window=128000,
        is_local=False,
        description="SIMULATED reference price. Not called by RAGOps.",
    ),
    ModelPricing(
        model_name="claude-haiku-4-5-20251001",
        provider="anthropic",
        input_cost_per_1k=0.00100,
        output_cost_per_1k=0.00500,
        context_window=200000,
        is_local=False,
        description="SIMULATED reference price. Not called by RAGOps.",
    ),
)

_PRICING_INDEX: Final[dict[str, ModelPricing]] = {
    p.model_name: p for p in DEFAULT_PRICING
}


def get_pricing(model_name: str) -> ModelPricing | None:
    return _PRICING_INDEX.get(model_name)


def calculate_cost(
    input_tokens: int,
    output_tokens: int,
    model_name: str,
    *,
    pricing: ModelPricing | None = None,
) -> float:
    """Estimated cost in USD for one generation.

    Returns exactly ``0.0`` for local models. For unknown models it falls back
    to zero rather than inventing a price — an unpriced model shows up in the
    cost/quality table as unpriced, which is honest.
    """
    if not settings.pricing_enabled:
        return 0.0
    entry = pricing or get_pricing(model_name)
    if entry is None or entry.is_local:
        return 0.0
    return round(
        (input_tokens / 1000.0) * entry.input_cost_per_1k
        + (output_tokens / 1000.0) * entry.output_cost_per_1k,
        8,
    )


def is_simulated(model_name: str) -> bool:
    """True when the cost shown for this model is a what-if figure."""
    entry = get_pricing(model_name)
    return entry is not None and not entry.is_local


def pricing_label(model_name: str) -> str:
    """Short provenance string the UI shows next to a dollar figure."""
    if not settings.pricing_enabled:
        return "Pricing disabled"
    entry = get_pricing(model_name)
    if entry is None:
        return "Unpriced model"
    if entry.is_local:
        return "Local inference — no API cost"
    return "Simulated list price"


# --------------------------------------------------------------------------
# Windowed aggregation helpers
# --------------------------------------------------------------------------

_WINDOW_SECONDS: Final[dict[str, int]] = {
    "1h": 3600,
    "24h": 86400,
    "7d": 604800,
    "30d": 2592000,
    "90d": 7776000,
}


def resolve_window(
    window: str | None,
    start: datetime | None = None,
    end: datetime | None = None,
) -> tuple[datetime, datetime, str]:
    """Resolve a window selector into concrete UTC bounds.

    ``window`` is one of ``1h|24h|7d|30d|90d``; supplying ``start``/``end``
    switches to a custom range and wins over ``window``. Returns
    ``(start, end, label)`` where ``label`` is what the UI echoes back.
    """
    end = end or datetime.now(tz=None).astimezone()
    if start is not None:
        return start, end, "custom"
    seconds = _WINDOW_SECONDS.get(window or "7d", _WINDOW_SECONDS["7d"])
    return end - timedelta(seconds=seconds), end, window or "7d"
