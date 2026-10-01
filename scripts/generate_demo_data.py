"""Generate a realistic synthetic workload so every dashboard page has data.

Run from the repository root:

    .venv/Scripts/python.exe scripts/generate_demo_data.py --traces 10000

Why this exists
---------------
An observability product is unusable with an empty database, but shipping a
dashboard whose numbers were typed in by hand would be worse -- it would make
the product look like it works before anything has been measured. So this
script *ingests synthetic traces through the real ingest service*, and every
number the dashboard then shows is an aggregate over rows that actually exist.

Consequences worth being explicit about:

* The traces are synthetic. The UI labels the demo application as such.
* Anomaly detection, regression detection, and the optimization engine still
  have real work to do, because the planted extremes below are planted in the
  *data*, not asserted in code. If the detectors stop firing, that is a finding.
* Nothing here is deterministic-by-constant: the distributions are sampled, the
  values vary per trace, and seed 7 only makes a run reproducible.

Usage
-----
    --traces N       number of traces to generate (default 10000)
    --seed N         RNG seed for reproducibility (default 7)
    --batch N        traces per ingest batch (default 250)
    --days N         spread the workload over N days (default 30)
    --reset          delete previously generated demo rows first
"""

from __future__ import annotations

import argparse
import asyncio
import math
import random
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

from sqlalchemy import delete, select  # noqa: E402

from app.core.database import dispose_engine, session_scope  # noqa: E402
from app.models import Trace  # noqa: E402
from app.repositories.common import get_or_create_application  # noqa: E402
from app.schemas.ingest import (  # noqa: E402
    GenerationIn,
    RetrievalIn,
    RetrievedDocumentIn,
    SpanIn,
    TraceCreate,
)
from app.telemetry.ingest_service import IngestService  # noqa: E402
from app.utils.tokens import estimate_tokens  # noqa: E402

DEMO_APPLICATION = "ragops-demo"
DEMO_ENVIRONMENT = "demo"

MODELS = [
    # (name, provider, context_window, share of traffic)
    ("qwen2.5:3b", "ollama", 8192, 0.55),
    ("qwen2.5:0.5b", "ollama", 4096, 0.25),
    ("llama3:latest", "ollama", 8192, 0.15),
    ("gpt-4o-mini", "openai", 128000, 0.05),
]

RETRIEVERS = ["hybrid_bm25_vector", "vector_faiss", "bm25", "hybrid_reranked"]

QUESTIONS = [
    "How do I reset my password?",
    "What is the refund policy for annual plans?",
    "How do I enable two-factor authentication?",
    "Which export formats do you support?",
    "How long does a data processing add-on take?",
    "Can I migrate my workspace to another region?",
    "How do I add a teammate to my team plan?",
    "What happens to my data if I cancel?",
    "How is usage calculated for API requests?",
    "Do you offer a self-hosted deployment?",
    "How do I rotate my API keys?",
    "What is the SLA for uptime?",
]

ANSWER_SHAPES = [
    "You can reset your password from Settings > Security. We email a reset link that expires in 30 minutes.",
    "Annual plans are refundable within 30 days of purchase. After that, we can offer account credit instead.",
    "Two-factor authentication is under Settings > Security > 2FA. We support TOTP apps and hardware keys.",
    "Exports are available as CSV, JSON, and Parquet on every plan. Scheduled exports are Pro and above.",
    "Data processing add-ons typically complete within 24 hours of purchase.",
    "Regional migration is available on Enterprise plans. Contact your account manager to start.",
    "Invite teammates from Workspace > Members. Each seat is billed separately.",
    "On cancellation your data is retained for 30 days, then permanently deleted.",
    "API usage is billed per 1K tokens, metered hourly and summed monthly.",
    "Self-hosted deployment is available on Enterprise. We provide a Docker Compose reference stack.",
    "API keys can be rotated from Settings > API. We keep the previous key valid for 24 hours.",
    "Our uptime SLA is 99.9% monthly on Enterprise, with service credits for missed targets.",
]

USERS = [f"user-{i:03d}" for i in range(40)]
CHUNKS = 134
SESSION_CHOICES = [f"sess-{i:04d}" for i in range(200)]


def _pick(rng: random.Random, weighted: list[tuple[str, str, int, float]]) -> tuple[str, str, int]:
    """Weighted choice over ``(name, provider, context_window, share)`` rows."""
    roll = rng.random()
    cumulative = 0.0
    for name, provider, ctx, share in weighted:
        cumulative += share
        if roll <= cumulative:
            return name, provider, ctx
    return weighted[-1][:3]


def _lognormal(rng: random.Random, median: float, sigma: float) -> float:
    """Latency/token draws: heavy right tail, which is what real traffic looks like."""
    return rng.lognormvariate(math.log(median), sigma)


def _plant_anomaly(rng: random.Random, kind: str) -> bool:
    """Mark a small fraction of traces as planted extremes for the detector.

    The signal is planted in the *data*; whether IsolationForest actually flags
    these rows is then a real question rather than an assertion. The rates are
    low enough that they sit in the tail rather than in a separate cluster --
    a flag rate that obvious would be detectable without any model at all.
    """
    return rng.random() < {"latency": 0.010, "tokens": 0.008}[kind]


def build_trace(
    rng: random.Random,
    index: int,
    total: int,
    start: datetime,
    window_seconds: float,
) -> TraceCreate:
    """Synthesise one trace across the span tree, retrieval, and LLM calls."""
    q_index = index % len(QUESTIONS)
    question = QUESTIONS[q_index]
    model_name, provider, ctx_window = _pick(rng, MODELS)
    retriever = rng.choice(RETRIEVERS)
    user_id = rng.choice(USERS)

    # Timestamps march forward across the requested window, so the last trace
    # lands at roughly `start + window_seconds` -- which is "now" when the
    # caller asked for `--days N` back from today. Spreading by a hardcoded 30
    # days instead would push a short run far into the future, where the
    # analytics window cannot see it and the trace looks like it never happened.
    offset = (index / max(total, 1)) * window_seconds
    jitter = rng.uniform(-1800, 1800)
    started = start + timedelta(seconds=max(0.0, offset + jitter))

    has_error = rng.random() < 0.018
    doc_count = rng.choice([3, 4, 5, 5, 6, 8])
    chunks = rng.sample(range(CHUNKS), min(doc_count, CHUNKS))
    # Duplicate retrieval is the measured waste the token-efficiency view shows.
    if rng.random() < 0.22:
        chunks = chunks + chunks[: rng.randint(1, 2)]

    documents = [
        RetrievedDocumentIn(
            document_id=f"chunk-{c:04d}",
            rank=rank,
            title=f"Knowledge base chunk {c}",
            content_preview=f"Excerpt {c} relevant to: {question[:48]}",
            bm25_score=round(rng.uniform(0.1, 4.0), 4),
            vector_score=round(rng.uniform(0.10, 0.95), 4),
            rerank_score=round(rng.uniform(-8.0, 9.0), 4),
            final_score=round(rng.uniform(0.05, 0.99), 4),
        )
        for rank, c in enumerate(chunks, start=1)
    ]

    retrieved_chars = sum(len(d.content_preview or "") for d in documents)
    context_tokens = retrieved_chars // 4

    prompt = f"{question}\n\nContext:\n" + "\n".join(
        (d.content_preview or "") for d in documents[:5]
    )

    latency_anomaly = _plant_anomaly(rng, "latency")
    token_anomaly = _plant_anomaly(rng, "tokens")

    prompt_tokens = estimate_tokens(prompt)
    if token_anomaly:
        prompt_tokens = int(prompt_tokens * rng.uniform(6.0, 12.0))

    answer = "" if has_error else ANSWER_SHAPES[q_index]
    completion_tokens = 0 if has_error else estimate_tokens(answer)
    if token_anomaly:
        completion_tokens = int(completion_tokens * rng.uniform(5.0, 9.0))

    generation_ms = _lognormal(rng, 780.0, 0.55)
    if latency_anomaly:
        generation_ms *= rng.uniform(8.0, 16.0)

    retrieval_ms = _lognormal(rng, 26.0, 0.42)
    rerank_ms = _lognormal(rng, 31.0, 0.38) if "rerank" in retriever else 0.0
    embed_ms = _lognormal(rng, 9.0, 0.3)

    total_ms = retrieval_ms + rerank_ms + embed_ms + generation_ms
    ended = started + timedelta(milliseconds=total_ms)

    top_score = round(min(0.999, max(0.05, rng.gauss(0.72, 0.13))), 4)
    if rng.random() < 0.03:  # a genuinely bad retrieval, as happens in practice
        top_score = round(rng.uniform(0.10, 0.35), 4)

    generations = [
        GenerationIn(
            model=model_name,
            provider=provider,
            prompt=prompt[:4000],
            completion="" if has_error else answer,
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            duration_ms=round(generation_ms, 2),
            time_to_first_token_ms=round(generation_ms * rng.uniform(0.10, 0.30), 2),
            temperature=0.1,
            max_tokens=min(ctx_window, 2048),
            status="error" if has_error else "success",
            # Sent explicitly because `llm_calls.created_at` is the event-time
            # column every token, cost and latency window buckets on. Left unset
            # it would fall back to the server's insert clock, which for a
            # backdated batch puts all 11,793 calls in the two minutes the
            # generator ran and flattens the trend chart to a single spike.
            timestamp=started,
            metadata={"simulated": True, "context_window": ctx_window},
        )
    ]

    # A few traces make a second model call, as real multi-step pipelines do.
    if rng.random() < 0.18:
        second_model, second_provider, _ = _pick(
            rng, [m for m in MODELS if m[0] != model_name]
        )
        second_ms = _lognormal(rng, 420.0, 0.5)
        followup = ANSWER_SHAPES[(q_index + 1) % len(ANSWER_SHAPES)]
        generations.append(
            GenerationIn(
                model=second_model,
                # The provider of the model actually chosen, not a hardcoded
                # "ollama". Labelling a gpt-4o-mini call as Ollama would put a
                # model/provider pair in the database that no real system ever
                # produces -- and the cost/quality chart would then have to
                # guess whether such a row is free or metered.
                provider=second_provider,
                prompt=answer[:2000],
                completion=followup,
                input_tokens=estimate_tokens(answer[:2000]),
                output_tokens=estimate_tokens(followup),
                duration_ms=round(second_ms, 2),
                status="success",
                timestamp=started + timedelta(milliseconds=total_ms),
                metadata={"simulated": True, "step": 2},
            )
        )
        total_ms += second_ms
        ended = started + timedelta(milliseconds=total_ms)

    is_agent = rng.random() < 0.09
    spans = [
        SpanIn(
            name="retrieve",
            kind="retrieval",
            duration_ms=round(retrieval_ms + rerank_ms + embed_ms, 2),
            status="success",
            attributes={"retriever": retriever, "top_k": doc_count, "top_score": top_score},
        ),
        SpanIn(
            name="generate",
            kind="llm",
            duration_ms=round(generation_ms, 2),
            status="error" if has_error else "success",
            attributes={"model": model_name, "provider": provider},
        ),
    ]
    if is_agent:
        steps = rng.randint(2, 5)
        spans.insert(
            0,
            SpanIn(
                name="agent_plan",
                kind="agent_step",
                duration_ms=round(total_ms * rng.uniform(0.05, 0.2), 2),
                status="success",
                attributes={"steps": steps},
            ),
        )

    metadata: dict[str, object] = {
        "simulated": True,
        "generator": "scripts/generate_demo_data.py",
        "planted": {
            "latency": latency_anomaly,
            "tokens": token_anomaly,
            "error": has_error,
        },
    }
    # NOTE: `token_efficiency` is deliberately *not* set here. IngestService owns
    # that metadata key, measures it from the token counts and duplicate
    # document ids it just wrote, and discards any caller-supplied value. A
    # generator that filled it in would be inventing a dashboard metric -- the
    # one thing this script must never do. The duplicates planted above are
    # therefore genuine measured input: the real scorer finds them itself.

    return TraceCreate(
        application=DEMO_APPLICATION,
        # Deterministic, not random: `trace_id` is the ingest idempotency key,
        # so re-running this script with the same --seed/--traces overwrites the
        # same rows rather than accumulating a second copy of the workload.
        trace_id=f"demo-{index:06d}",
        user_id=user_id,
        session_id=rng.choice(SESSION_CHOICES),
        kind="agent" if is_agent else "rag",
        name="support_bot_query",
        start_time=started,
        end_time=ended,
        duration_ms=round(total_ms, 2),
        input_text=question,
        output_text=answer or None,
        error=(
            rng.choice(
                [
                    "upstream_model_timeout",
                    "retrieval_index_unavailable",
                    "context_window_exceeded",
                ]
            )
            if has_error
            else None
        ),
        status="error" if has_error else "success",
        agent_name="support_agent" if is_agent else None,
        agent_iterations=rng.randint(1, 4) if is_agent else 0,
        context_tokens=context_tokens,
        tags=["demo", "synthetic"],
        metadata=metadata,
        spans=spans,
        retrievals=[
            RetrievalIn(
                query=question,
                retriever=retriever,
                top_k=doc_count,
                duration_ms=round(retrieval_ms, 2),
                documents=documents,
                # Same reason as the generation timestamps: this is what puts
                # the row in the right day bucket for retrieval analytics.
                timestamp=started,
                configuration={
                    "top_k": doc_count,
                    "hybrid_rrf_k": 60,
                    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                    "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                    "simulated": True,
                },
            )
        ],
        generations=generations,
    )


async def reset_demo_data(session, application_id: uuid.UUID) -> int:
    """Delete traces previously generated for the demo application."""
    rows = await session.execute(
        select(Trace.id).where(Trace.application_id == application_id)
    )
    ids = [r[0] for r in rows]
    if not ids:
        return 0
    await session.execute(delete(Trace).where(Trace.id.in_(ids)))
    await session.commit()
    return len(ids)


async def main() -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic RAGOps traces.")
    parser.add_argument("--traces", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch", type=int, default=250, choices=range(1, 501))
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument(
        "--reset",
        action="store_true",
        help=(
            "Delete every existing trace for the demo application first. This "
            "is destructive and is only needed when traces generated by an "
            "older run cannot be overwritten -- for example rows written "
            "before trace ids became deterministic. Normal re-runs need it "
            "not: the same --seed and --traces overwrite the same rows."
        ),
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    start = datetime.now(timezone.utc) - timedelta(days=args.days)
    window_seconds = float(args.days) * 24 * 3600
    service = IngestService()

    async with session_scope() as session:
        application = await get_or_create_application(
            session,
            DEMO_APPLICATION,
            description=(
                "Synthetic demonstration workload. Every number shown for this "
                "application is an aggregate over generated rows, not a "
                "hardcoded metric."
            ),
            environment=DEMO_ENVIRONMENT,
        )
        await session.commit()

        if args.reset:
            removed = await reset_demo_data(session, application.id)
            print(f"Removed {removed} existing demo trace(s).")

    ingested = 0
    started = asyncio.get_running_loop().time()

    for offset in range(0, args.traces, args.batch):
        size = min(args.batch, args.traces - offset)
        payloads = [
            build_trace(rng, offset + i, args.traces, start, window_seconds)
            for i in range(size)
        ]
        async with session_scope() as session:
            # The service takes the list itself. `BatchIngest` is the HTTP
            # request-body schema, not a service argument -- iterating one
            # yields (field, value) tuples, not traces.
            response = await service.ingest_batch(session, payloads)
        ingested += response.accepted
        if response.errors:
            # A silent partial write would make the summary below a lie.
            print(f"    {len(response.errors)} rejected: {response.errors[0]}")

        elapsed = asyncio.get_running_loop().time() - started
        rate = ingested / elapsed if elapsed else 0.0
        print(
            f"  {ingested:>6,}/{args.traces:,} traces  "
            f"({rate:.0f}/s, {elapsed:.1f}s elapsed)",
            flush=True,
        )

    await dispose_engine()

    print(f"\nDone. {ingested:,} traces ingested into application '{DEMO_APPLICATION}'.")
    print("All dashboard figures are now aggregates over these rows.")
    print("Run scripts/seed_models.py first if you want per-model cost rollups.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
