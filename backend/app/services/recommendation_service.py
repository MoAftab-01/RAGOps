"""Evidence-gated optimisation recommendations.

The rules here are deterministic functions of recorded measurements. That is a
design constraint, not a shortcut: a recommendation is a *claim* stored
deliberately in the database, and the only thing keeping that claim honest is
that it is mechanically derived from a number the reader can go and check. An
LLM asked "what should I optimise?" would produce fluent suggestions with no
traceable derivation, which is precisely the failure mode this product exists to
catch in other people's systems.

Two rules govern every recommendation emitted here:

1. **No measurement, no recommendation.** Each rule computes a statistic, then
   decides whether that statistic crosses a threshold. Below the threshold the
   rule returns nothing at all — not a low-priority suggestion, not a hedged one.
   A recommendation nobody can act on is still noise on the page.
2. **The evidence names the measurement.** ``evidence`` carries the metric name,
   its value, the threshold it crossed, and the number of samples it was
   computed over. If a statistic cannot be computed, the rule does not fire.
   The service asserts this in :func:`_build` rather than trusting each rule to
   remember.

Saving money is *bounded*, never promised. Every rule that could imply a saving
states the ceiling the data supports ("duplicates are 12% of input tokens, so at
most 12% of input-token cost is addressable") and never multiplies that into a
dollar figure the platform cannot substantiate.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models import LLMCall, OptimizationRecommendation, Trace
from app.repositories.common import window_predicates
from app.services.window import AnalyticsScope

logger = get_logger(__name__)

__all__ = ["RecommendationEngine", "RecommendationResult"]


#: Fraction of a metric above which a rule fires. Each is chosen to be well clear
#: of the noise floor of a small demo workload rather than tuned to make any
#: particular number appear.
TOKEN_WASTE_THRESHOLD = 0.05
CACHE_HIT_THRESHOLD = 0.20
TOP_K_BLOAT_FACTOR = 2.0
SMALL_MODEL_INPUT_TOKENS = 512
LATENCY_P95_MULTIPLE = 3.0
ERROR_RATE_THRESHOLD = 0.02
UNUSED_CONTEXT_FRACTION = 0.60
#: Below this mean context size, trimming context is not worth recommending.
#: Citation coverage is measured against short document previews, so a small
#: pipeline can score badly on it while the real prompt is a few hundred tokens.
MIN_MEANINGFUL_CONTEXT_TOKENS = 500.0


@dataclass(frozen=True)
class RecommendationResult:
    """What a generation pass produced."""

    recommendations: list[OptimizationRecommendation]
    #: Per-rule outcome, including the rules that did not fire. A generation run
    #: that reports "0 recommendations" and does not say which rules ran and what
    #: they measured is indistinguishable from an engine that found nothing, and
    #: from one that was never called.
    rules_evaluated: list[dict[str, Any]]

    @property
    def generated(self) -> int:
        return len(self.recommendations)


class RecommendationEngine:
    """Deterministic rule set over recorded telemetry."""

    def __init__(self) -> None:
        self._rules = (
            self._rule_duplicate_context,
            self._rule_repeated_prompts,
            self._rule_oversized_context,
            self._rule_small_model_long_prompts,
            self._rule_slow_tail,
            self._rule_error_rate,
        )

    async def generate(
        self,
        session: AsyncSession,
        scope: AnalyticsScope,
        *,
        min_severity: float = 0.0,
        persist: bool = True,
    ) -> RecommendationResult:
        """Run every rule over the window and return the ones that fired."""
        produced: list[OptimizationRecommendation] = []
        evaluated: list[dict[str, Any]] = []

        for rule in self._rules:
            try:
                recommendation, observation = await rule(session, scope)
            except Exception:  # pragma: no cover - a broken rule must not
                # take the whole pass down and lose the rules that did work.
                # The failure is logged with the rule name so it is visible,
                # and the rule is recorded as errored rather than silently
                # skipped.
                logger.warning(
                    "recommendations.rule_failed", rule=rule.__name__, exc_info=True
                )
                evaluated.append({"rule": rule.__name__, "fired": False, "error": True})
                continue

            evaluated.append(observation)
            if recommendation is None:
                continue
            if recommendation.severity_score < min_severity:
                observation["suppressed_by_min_severity"] = True
                continue
            produced.append(recommendation)

        if persist and produced:
            for recommendation in produced:
                session.add(recommendation)
            await session.flush()
            # `id` is a Python-side default, so the flush populated it, but
            # `created_at` is a *server* default — it does not exist until
            # PostgreSQL has seen the INSERT. Without this refresh the route
            # serialises the row it just created and gets None for both, which
            # the response schema correctly refuses to accept. The reloaded row
            # is what was actually stored, which is also the right thing to
            # hand back.
            for recommendation in produced:
                await session.refresh(recommendation)

        logger.info(
            "recommendations.generated",
            application=scope.application_name,
            window=scope.window,
            rules=len(self._rules),
            generated=len(produced),
            persisted=persist,
        )
        return RecommendationResult(recommendations=produced, rules_evaluated=evaluated)

    # ------------------------------------------------------------------
    # Rules. Each returns (recommendation | None, observation).
    # ------------------------------------------------------------------

    async def _rule_duplicate_context(
        self, session: AsyncSession, scope: AnalyticsScope
    ) -> tuple[OptimizationRecommendation | None, dict[str, Any]]:
        """Flag retrieved-context duplication — the clearest waste signal there is.

        Evidence is the measured duplicate-document ratio, which
        :class:`~app.telemetry.ingest_service.IngestService` computes at ingest
        from the document ids actually written. It is a measurement, not an
        estimate: the same documents were observed arriving twice.
        """
        fraction, samples = await _jsonb_mean(
            session,
            scope,
            "retrieval_analysis",
            "duplicate_ratio",
        )
        observation: dict[str, Any] = {
            "rule": "duplicate_context",
            "metric": "mean duplicate_document_ratio",
            "value": fraction,
            "samples": samples,
            "threshold": TOKEN_WASTE_THRESHOLD,
            "fired": False,
        }
        if fraction is None or samples < _MIN_SAMPLES or fraction < TOKEN_WASTE_THRESHOLD:
            return None, observation

        observation["fired"] = True
        observation["addressable_token_fraction"] = round(fraction, 4)
        recommendation = _build(
            application_id=scope.application_id,
            category="token_waste",
            title="Retrieved context contains duplicate documents",
            rationale=(
                f"{fraction:.1%} of retrieved documents in this window were "
                f"duplicates, measured across {samples} scored requests. The same "
                f"passage was sent to the model more than once in a single "
                f"prompt."
            ),
            recommendation=(
                "Deduplicate retrieved documents by document id before assembling "
                "the context window, and cap top_k at the point where recall stops "
                f"improving (at most {fraction:.0%} of input tokens are recoverable "
                "this way)."
            ),
            # Severity scales with how much of the context is redundant. A ratio
            # of 0.06 is a rounding error; 0.6 means most of the prompt is repeat.
            severity=min(1.0, fraction),
            priority=_priority(fraction),
            confidence=min(1.0, 0.5 + samples / 2000),
            evidence={
                "metric": "mean duplicate_document_ratio",
                "value": round(fraction, 4),
                "threshold": TOKEN_WASTE_THRESHOLD,
                "num_samples": samples,
                "source": "traces.metadata.retrieval_analysis.duplicate_ratio",
            },
            metrics={"duplicate_ratio": round(fraction, 4), "num_samples": samples},
        )
        return recommendation, observation

    async def _rule_repeated_prompts(
        self, session: AsyncSession, scope: AnalyticsScope
    ) -> tuple[OptimizationRecommendation | None, dict[str, Any]]:
        """Flag a repeated prefix in prompts — the thing prompt caching exists for.

        The prefix is a real measurement: a byte-for-byte shared head across
        requests, not a guess that "these requests look similar".
        """
        fraction, samples, example = await _repeated_prefix_fraction(session, scope)
        observation: dict[str, Any] = {
            "rule": "repeated_prompt_prefix",
            "metric": "share of requests sharing a 200-char prompt prefix",
            "value": fraction,
            "samples": samples,
            "threshold": CACHE_HIT_THRESHOLD,
            "fired": False,
        }
        if fraction is None or samples < _MIN_SAMPLES or fraction < CACHE_HIT_THRESHOLD:
            return None, observation

        observation["fired"] = True
        observation["example_prefix"] = example
        recommendation = _build(
            application_id=scope.application_id,
            category="token_cost",
            title="A large share of requests share an identical prompt prefix",
            rationale=(
                f"{fraction:.1%} of {samples} requests in this window begin with a "
                f"byte-identical 200-character prefix. That prefix is re-sent and "
                f"re-billed on every one of them."
            ),
            recommendation=(
                "Hoist the invariant prefix (system prompt, few-shot examples, "
                "tool definitions) to the front of the prompt and enable the "
                "provider's prompt cache, so the shared head is billed once."
            ),
            severity=min(1.0, fraction),
            priority=_priority(fraction),
            confidence=min(1.0, 0.5 + samples / 2000),
            evidence={
                "metric": "share of requests with a shared 200-char prompt prefix",
                "value": round(fraction, 4),
                "threshold": CACHE_HIT_THRESHOLD,
                "num_samples": samples,
                "example_prefix": example,
                "source": "llm_calls.prompt",
            },
            metrics={"shared_prefix_fraction": round(fraction, 4), "num_samples": samples},
        )
        return recommendation, observation

    async def _rule_oversized_context(
        self, session: AsyncSession, scope: AnalyticsScope
    ) -> tuple[OptimizationRecommendation | None, dict[str, Any]]:
        """Flag context that is large but weakly referenced by the answer.

        Low citation coverage against a large context means the retrieval stage
        is sending passages the model does not use. The suggested cut is an
        upper bound derived from the measured unused share, not a guess.

        The rule stays silent unless the context is also *big*. Citation
        coverage is measured against the retrieved documents' previews, which
        are short by construction, so a pipeline that retrieves a handful of
        short previews can score near-zero coverage while the real prompt is a
        few hundred tokens — and telling someone to cut their context to save
        money, off a 75-token average, would be noise dressed as advice.
        """
        stats = await _context_citation_stats(session, scope)
        observation: dict[str, Any] = {
            "rule": "unused_context",
            "metric": "share of retrieved characters not cited by the answer",
            "value": stats.get("unused_fraction"),
            "samples": stats.get("samples"),
            "avg_context_tokens": stats.get("avg_context_tokens"),
            "threshold": UNUSED_CONTEXT_FRACTION,
            "fired": False,
        }
        if stats.get("unused_fraction") is None or stats.get("samples", 0) < _MIN_SAMPLES:
            return None, observation

        if stats["avg_context_tokens"] < MIN_MEANINGFUL_CONTEXT_TOKENS:
            # Recorded as a non-firing observation with the reason, not dropped.
            # "The rule ran and had nothing to say, because X" is different
            # from "the rule did not run", and only the first is a finding.
            observation["reason"] = (
                f"mean context is only {stats['avg_context_tokens']:.0f} tokens; "
                f"below the {MIN_MEANINGFUL_CONTEXT_TOKENS}-token floor at which "
                f"trimming context is worth recommending"
            )
            return None, observation
        if stats["unused_fraction"] < UNUSED_CONTEXT_FRACTION:
            return None, observation

        observation["fired"] = True
        recommendation = _build(
            application_id=scope.application_id,
            category="retrieval",
            title="Most retrieved context is not used by the answer",
            rationale=(
                f"Only {1 - stats['unused_fraction']:.0%} of retrieved text was "
                f"referenced by the generated answer across {stats['samples']} "
                f"scored requests, while average context was "
                f"{stats['avg_context_tokens']:.0f} tokens. The retrieval stage is "
                f"paying to send text the model does not cite."
            ),
            recommendation=(
                f"Reduce top_k or tighten the reranker's cutoff. Lowering k until "
                f"citation coverage stops falling would remove up to "
                f"{stats['unused_fraction']:.0%} of context tokens; measure the "
                f"recall change on a labelled set before shipping it."
            ),
            severity=min(1.0, stats["unused_fraction"]),
            priority=_priority(stats["unused_fraction"]),
            confidence=min(1.0, 0.5 + stats["samples"] / 2000),
            evidence={
                "metric": "unused retrieved-text fraction",
                "value": round(stats["unused_fraction"], 4),
                "threshold": UNUSED_CONTEXT_FRACTION,
                "num_samples": stats["samples"],
                "avg_context_tokens": round(stats["avg_context_tokens"], 1),
                "source": "traces.output_text vs retrieved_documents.content_preview",
                "caveat": (
                    "Lexical word-overlap proxy, not a semantic citation check: an "
                    "answer that paraphrases a passage shares few words with it and "
                    "is scored as uncited."
                ),
            },
            metrics={
                "unused_fraction": round(stats["unused_fraction"], 4),
                "avg_context_tokens": round(stats["avg_context_tokens"], 1),
                "num_samples": stats["samples"],
            },
        )
        return recommendation, observation

    async def _rule_small_model_long_prompts(
        self, session: AsyncSession, scope: AnalyticsScope
    ) -> tuple[OptimizationRecommendation | None, dict[str, Any]]:
        """Flag large prompts sent to a model that cannot use them well.

        Both halves are measured: the mean input size of the calls to the
        cheapest model, and that model's context window from the registry. A
        prompt that fills the window of a 0.5B model is spending latency and
        memory for text the model has no capacity to attend to.
        """
        model, avg_input, context_window, calls = await _smallest_model_usage(
            session, scope
        )
        observation: dict[str, Any] = {
            "rule": "oversized_prompt_for_small_model",
            "metric": "mean input tokens vs smallest-model context window",
            "value": avg_input,
            "model": model,
            "context_window": context_window,
            "samples": calls,
            "threshold": SMALL_MODEL_INPUT_TOKENS,
            "fired": False,
        }
        if (
            model is None
            or calls < _MIN_SAMPLES
            or context_window is None
            or avg_input < SMALL_MODEL_INPUT_TOKENS
        ):
            return None, observation

        # Only meaningful when the prompts are actually a large share of the
        # window; a 600-token prompt into a 4096-token model is unremarkable.
        if avg_input < context_window * 0.5:
            observation["reason"] = "prompt is under half the model's context window"
            return None, observation

        fill = min(1.0, avg_input / context_window)
        observation["fired"] = True
        observation["window_fill_fraction"] = round(fill, 4)
        recommendation = _build(
            application_id=scope.application_id,
            category="model_selection",
            title=f"Prompts are filling the context window of {model}",
            rationale=(
                f"{model} received a mean of {avg_input:.0f} input tokens across "
                f"{calls} calls — {fill:.0%} of its {context_window}-token window. "
                f"A model that small attends poorly over long contexts, so the "
                f"extra text costs latency without reliably being used."
            ),
            recommendation=(
                f"Either shorten what is sent to {model} (retrieve fewer, more "
                f"relevant passages) or route these long-context requests to a "
                f"larger local model. Re-run the retrieval evaluation at the "
                f"reduced k before switching."
            ),
            severity=fill,
            priority=_priority(fill),
            confidence=min(1.0, 0.5 + calls / 2000),
            evidence={
                "metric": "mean input tokens per call",
                "value": round(avg_input, 1),
                "model": model,
                "context_window": context_window,
                "window_fill_fraction": round(fill, 4),
                "num_calls": calls,
                "source": "llm_calls.input_tokens joined to the models registry",
            },
            metrics={
                "mean_input_tokens": round(avg_input, 1),
                "context_window": context_window,
                "num_calls": calls,
            },
        )
        return recommendation, observation

    async def _rule_slow_tail(
        self, session: AsyncSession, scope: AnalyticsScope
    ) -> tuple[OptimizationRecommendation | None, dict[str, Any]]:
        """Flag a long tail well beyond the median, from the percentile columns.

        Uses P95/P50 rather than "is it slow": a healthy service has a tail, and
        only a tail several times the median indicates something to look at.
        """
        p50, p95, samples = await _latency_percentiles(session, scope)
        observation: dict[str, Any] = {
            "rule": "latency_tail",
            "metric": "p95 / p50 latency ratio",
            "value": (p95 / p50) if (p50 and p95) else None,
            "p50_ms": p50,
            "p95_ms": p95,
            "samples": samples,
            "threshold": LATENCY_P95_MULTIPLE,
            "fired": False,
        }
        if p50 is None or p95 is None or samples < _MIN_SAMPLES:
            return None, observation
        if p50 <= 0 or p95 / p50 < LATENCY_P95_MULTIPLE:
            return None, observation

        ratio = p95 / p50
        observation["fired"] = True
        # Severity is the *excess* over the threshold, so a service that is
        # merely slow does not outrank one whose tail has genuinely detached.
        severity = min(1.0, (ratio - LATENCY_P95_MULTIPLE) / LATENCY_P95_MULTIPLE)
        recommendation = _build(
            application_id=scope.application_id,
            category="latency",
            title="The latency tail has detached from the median",
            rationale=(
                f"P50 is {p50:.0f}ms but P95 is {p95:.0f}ms — {ratio:.1f}x the "
                f"median across {samples} requests. A tail this far out usually "
                f"means a slow path rather than uniform slowness, which is the "
                f"cheap kind of latency to fix."
            ),
            recommendation=(
                "Segment the slowest requests by model and retrieval stage to find "
                "which one produces the tail; the per-stage and per-model latency "
                "breakdowns on this page carry the same window."
            ),
            severity=severity,
            priority=_priority(ratio / LATENCY_P95_MULTIPLE),
            confidence=min(1.0, 0.5 + samples / 2000),
            evidence={
                "metric": "p95/p50 latency ratio",
                "value": round(ratio, 4),
                "threshold": LATENCY_P95_MULTIPLE,
                "p50_ms": round(p50, 2),
                "p95_ms": round(p95, 2),
                "num_samples": samples,
                "source": "traces.duration_ms",
            },
            metrics={"p50_ms": round(p50, 2), "p95_ms": round(p95, 2), "num_samples": samples},
        )
        return recommendation, observation

    async def _rule_error_rate(
        self, session: AsyncSession, scope: AnalyticsScope
    ) -> tuple[OptimizationRecommendation | None, dict[str, Any]]:
        """Flag a non-trivial error rate, naming the errors that caused it.

        The recommendation does not assert a cause — it reports the error
        breakdown the data shows and points at it. Which of those is *why* is a
        question for whoever owns the system.
        """
        total, errors, breakdown = await _error_breakdown(session, scope)
        rate = (errors / total) if total else None
        observation: dict[str, Any] = {
            "rule": "error_rate",
            "metric": "error rate",
            "value": rate,
            "samples": total,
            "threshold": ERROR_RATE_THRESHOLD,
            "fired": False,
        }
        if rate is None or total < _MIN_SAMPLES or rate < ERROR_RATE_THRESHOLD:
            return None, observation

        observation["fired"] = True
        observation["error_breakdown"] = breakdown
        top = list(breakdown.items())[:3]
        detail = ", ".join(f"{name} ({count})" for name, count in top) or "unclassified"
        recommendation = _build(
            application_id=scope.application_id,
            category="reliability",
            title=f"{rate:.1%} of requests are failing",
            rationale=(
                f"{errors} of {total} requests in this window ended in an error. "
                f"The recorded error values were: {detail}. This states the rate "
                f"and the distribution, not a cause — the data cannot distinguish "
                f"which factor is responsible."
            ),
            recommendation=(
                "Group the failing requests on the Traces page by error value and "
                "compare their latency and token counts against the successful "
                "ones. The two distributions usually localise the failure to a "
                "stage."
            ),
            severity=min(1.0, rate * 5),
            priority=_priority(rate * 5),
            confidence=min(1.0, 0.5 + total / 2000),
            evidence={
                "metric": "error rate",
                "value": round(rate, 6),
                "threshold": ERROR_RATE_THRESHOLD,
                "num_samples": total,
                "num_errors": errors,
                "error_breakdown": dict(breakdown),
                "source": "traces.status / traces.error",
            },
            metrics={
                "error_rate": round(rate, 6),
                "num_errors": errors,
                "num_samples": total,
            },
        )
        return recommendation, observation


#: Below this many requests a rate is a handful of rows, and a rule that fires on
#: a handful of rows produces a confident recommendation about nothing.
_MIN_SAMPLES = 30


def _build(
    *,
    application_id: uuid.UUID | None,
    category: str,
    title: str,
    rationale: str,
    recommendation: str,
    severity: float,
    priority: str,
    confidence: float,
    evidence: dict[str, Any],
    metrics: dict[str, Any] | None = None,
) -> OptimizationRecommendation:
    """Construct an unsaved recommendation row.

    The evidence assertion is the whole point of this module. A recommendation
    that cannot name the measurement behind it is a guess with a confidence
    score attached, and the model allows ``evidence`` to be null — so it is
    checked here rather than trusted from each of the six call sites.
    """
    if not evidence or evidence.get("value") is None:
        raise ValueError(
            f"refusing to build recommendation {title!r} without a measured value"
        )
    return OptimizationRecommendation(
        application_id=application_id,
        category=category,
        title=title,
        rationale=rationale,
        recommendation=recommendation,
        priority=priority,
        severity_score=round(float(severity), 4),
        evidence=evidence,
        metrics=metrics,
        status="open",
        confidence=round(float(confidence), 4),
    )


def _priority(severity: float) -> str:
    if severity >= 0.5:
        return "high"
    if severity >= 0.2:
        return "medium"
    return "low"


def _scope_filters(scope: AnalyticsScope) -> list[Any]:
    filters: list[Any] = [
        Trace.start_time >= scope.start,
        Trace.start_time <= scope.end,
        *window_predicates(scope.window_filter(), Trace.application_id),
    ]
    return filters


async def _jsonb_mean(
    session: AsyncSession, scope: AnalyticsScope, block: str, key: str
) -> tuple[float | None, int]:
    """Mean of a numeric value inside a JSONB metadata block.

    Computed in SQL with an explicit float cast: PostgreSQL has no ``avg()`` over
    text, and letting it implicitly coerce would be a guess. ``NULL`` rows --
    traces ingested before this analysis existed -- are excluded from the mean
    rather than counted as zero, so the figure describes the traces that
    actually carry the measurement.
    """
    from sqlalchemy import cast, Float, func

    value = cast(
        func.jsonb_extract_path_text(Trace.extra_metadata, block, key), Float
    )
    stmt = select(
        func.avg(value), func.count(value)
    ).where(*_scope_filters(scope), value.isnot(None))
    row = (await session.execute(stmt)).one()
    average, samples = row[0], row[1]
    return (None if average is None else float(average), int(samples or 0))


async def _repeated_prefix_fraction(
    session: AsyncSession, scope: AnalyticsScope
) -> tuple[float | None, int, str | None]:
    """Share of prompts sharing a prefix, with the most common prefix as example.

    The prefix is hashed to keep the query cheap: grouping 10k prompts by a
    200-character slice is a sort over a megabyte of text, and the group *count*
    is all the rule needs. The example prefix is the text of the largest group,
    so the recommendation can show what it actually matched rather than a hash.
    """
    from sqlalchemy import String, cast, func

    prefix = func.left(cast(LLMCall.prompt, String), 200)
    stmt = (
        select(prefix.label("p"), func.count().label("n"))
        .where(*_llm_filters(scope), LLMCall.prompt.isnot(None))
        .group_by(prefix)
        .order_by(func.count().desc())
        .limit(20)
    )
    groups = (await session.execute(stmt)).all()
    total_stmt = select(func.count()).where(
        *_llm_filters(scope), LLMCall.prompt.isnot(None)
    )
    total = int((await session.execute(total_stmt)).scalar() or 0)

    if not groups or total == 0:
        return None, 0, None
    top_prefix, top_count = groups[0][0], int(groups[0][1])
    return top_count / total, total, top_prefix


def _llm_filters(scope: AnalyticsScope) -> list[Any]:
    """Window filters for llm_calls.

    The window is applied to the *call* timestamp rather than the trace's, so a
    rule about model usage sees the calls that actually happened inside the
    window instead of the calls belonging to traces that started there.

    The tenant predicate is the one from the shared builder, keyed on
    ``Trace.application_id`` rather than ``LLMCall.application_id``. That is
    deliberate: the call's own column is nullable, and an unattached call has no
    application to compare against, whereas its trace always does. Filtering
    the same way as ``_scope_filters`` keeps the two rule families answering
    the same question about who the data belongs to.
    """
    filters: list[Any] = [
        LLMCall.created_at >= scope.start,
        LLMCall.created_at <= scope.end,
        LLMCall.trace_id.in_(
            select(Trace.id).where(
                *window_predicates(scope.window_filter(), Trace.application_id)
            )
        ),
    ]
    return filters


async def _context_citation_stats(
    session: AsyncSession, scope: AnalyticsScope
) -> dict[str, Any]:
    """Share of retrieved text that the answer actually references.

    Measured by token overlap between the answer and each retrieved document's
    text. This is a lexical proxy, not a semantic citation check — an answer can
    paraphrase a passage without sharing words, and this would score that as
    unused. It is used only to raise a rule, never to assert a saving, and the
    rule's text says it is a proxy.
    """
    from app.models import RetrievalCall, RetrievedDocument

    # The document table hangs off the retrieval call, and the retrieval call
    # hangs off the trace, so the join needs both hops to get back to a trace's
    # start_time. There is no direct trace_id on a document.
    stmt = (
        select(Trace.output_text, RetrievedDocument.content_preview)
        .select_from(RetrievedDocument)
        .join(RetrievalCall, RetrievalCall.id == RetrievedDocument.retrieval_call_id)
        .join(Trace, Trace.id == RetrievalCall.trace_id)
        .where(
            *_scope_filters(scope),
            Trace.output_text.isnot(None),
            RetrievedDocument.content_preview.isnot(None),
        )
        .limit(20_000)
    )
    rows = (await session.execute(stmt)).all()

    total_chars = 0
    cited_chars = 0
    for answer, preview in rows:
        if not answer or not preview:
            continue
        total_chars += len(preview)
        # Word-level overlap: a cited passage shares most of its content words
        # with the answer. The 0.5 threshold is the point where "this text was
        # used" is a better description than "this text was not".
        answer_words = {word for word in answer.lower().split() if len(word) > 3}
        preview_words = {word for word in preview.lower().split() if len(word) > 3}
        if not preview_words:
            continue
        overlap = len(answer_words & preview_words) / len(preview_words)
        if overlap >= 0.5:
            cited_chars += len(preview)

    if total_chars == 0:
        return {"unused_fraction": None, "samples": 0, "avg_context_tokens": 0.0}

    unused = 1.0 - (cited_chars / total_chars)
    avg_context = await _mean_column(session, scope, Trace.context_tokens)
    return {
        "unused_fraction": unused,
        "samples": len(rows),
        "avg_context_tokens": avg_context or 0.0,
    }


async def _mean_column(
    session: AsyncSession, scope: AnalyticsScope, column: Any
) -> float | None:
    stmt = select(func.avg(column)).where(*_scope_filters(scope), column.isnot(None))
    value = (await session.execute(stmt)).scalar()
    return None if value is None else float(value)


async def _smallest_model_usage(
    session: AsyncSession, scope: AnalyticsScope
) -> tuple[str | None, float, int | None, int]:
    """Mean input tokens and context window for the smallest model in use.

    "Smallest" is by registered context window, not by name, so the rule does
    not encode an assumption about which model happens to be called ``0.5b``.
    Models that were never registered have no known window and are excluded --
    guessing one would be exactly the kind of claim this module avoids.
    """
    from app.models import Model

    rows = (
        await session.execute(
            select(
                LLMCall.model_name,
                func.min(Model.context_window).label("window"),
                func.avg(LLMCall.input_tokens).label("avg_input"),
                func.count().label("calls"),
            )
            .join(Model, Model.name == LLMCall.model_name)
            .where(*_llm_filters(scope), LLMCall.input_tokens.isnot(None))
            .group_by(LLMCall.model_name)
        )
    ).all()

    usable = [r for r in rows if r.window is not None and r.avg_input is not None]
    if not usable:
        return None, 0.0, None, 0
    smallest = min(usable, key=lambda r: r.window)
    return str(smallest.model_name), float(smallest.avg_input), int(smallest.window), int(smallest.calls)


async def _latency_percentiles(
    session: AsyncSession, scope: AnalyticsScope
) -> tuple[float | None, float | None, int]:
    """P50 and P95 duration, plus the sample count, in one pass.

    ``percentile_cont`` returns an interval when the requested percentile falls
    between two rows, hence the cast before it is read as a number.
    """
    from sqlalchemy import Float, cast, func

    duration = Trace.duration_ms
    # The window filters are added to the *aggregate* rather than to a WHERE,
    # because an aggregate with no GROUP BY and a WHERE is the shape PostgreSQL
    # rejects: it has to be a HAVING, or the filters have to be attached to the
    # FROM subquery below.
    filters = _scope_filters(scope)
    scoped = select(Trace.duration_ms).where(*filters, duration.isnot(None)).subquery()
    stmt = select(
        cast(func.percentile_cont(0.5).within_group(scoped.c.duration_ms), Float).label("p50"),
        cast(
            func.percentile_cont(0.95).within_group(scoped.c.duration_ms), Float
        ).label("p95"),
        func.count().label("n"),
    )
    row = (await session.execute(stmt)).one()
    return (
        None if row[0] is None else float(row[0]),
        None if row[1] is None else float(row[1]),
        int(row[2] or 0),
    )


async def _error_breakdown(
    session: AsyncSession, scope: AnalyticsScope
) -> tuple[int, int, list[tuple[str, int]]]:
    """Total traces, errored traces, and the error values ranked by frequency.

    A failed trace with no recorded error string is grouped as ``unclassified``
    rather than dropped, so the counts still add up to the error total.
    """
    filters = _scope_filters(scope)
    total = int(
        (
            await session.execute(select(func.count()).select_from(Trace).where(*filters))
        ).scalar()
        or 0
    )
    errored_filter = [*filters, Trace.status == "error"]
    errors = int(
        (
            await session.execute(
                select(func.count()).select_from(Trace).where(*errored_filter)
            )
        ).scalar()
        or 0
    )
    # The grouped expression has to be the *same* object in SELECT and GROUP BY.
    # Repeating the func.coalesce() call makes SQLAlchemy emit two distinct
    # expressions and PostgreSQL rejects the second one as not being in the
    # GROUP BY, so it is bound once and reused.
    error_value = func.coalesce(Trace.error, "unclassified")
    breakdown_rows = (
        await session.execute(
            select(error_value, func.count())
            .where(*errored_filter)
            .group_by(error_value)
            .order_by(func.count().desc())
        )
    ).all()
    breakdown = [(str(name), int(count)) for name, count in breakdown_rows]
    return total, errors, breakdown
