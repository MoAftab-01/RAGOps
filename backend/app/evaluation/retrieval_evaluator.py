"""Run the real retriever over a labelled dataset and record what happened.

Two properties make this module different from a benchmark script, and both
exist because RAGOps has to be able to answer "did that change help or hurt?"
weeks later:

1. **The real retriever runs.** Scores come from
   :class:`app.ml.hybrid.HybridRetriever` over the actual knowledge base —
   BM25, vector search, RRF fusion, optional cross-encoder rerank. Nothing here
   is reimplemented, so an evaluation measures the system users query rather
   than a reimplementation of it. If the index is empty the run *stops* and says
   so; a report of 0.0 recall against no index would be a measurement of
   nothing.
2. **The configuration is recorded with the results.** ``k``, the RRF constant,
   the embedding and reranker checkpoints, the chunking parameters and the
   corpus size all land in ``EvaluationRun.config``. Metrics without their
   configuration are a number with no explanation, and an unexplained number
   cannot be compared against another unexplained number.

Scoring itself is delegated wholesale to :mod:`app.evaluation.metrics`, which is
pure and deterministic. This module contributes no arithmetic of its own beyond
counting, so there is no second implementation of Precision/Recall/MRR/NDCG that
could disagree with the first.

The dataset schema is read from the file rather than assumed. The loader accepts
the ``relevant_document_ids`` spelling actually used by
``evaluation/datasets/retrieval_eval.jsonl`` and the ``relevant_documents``
spelling of ``app.schemas.evaluation.EvalExample``, and it reads two facts the
contract does not mention: ``no_answer`` rows, whose relevance set is
deliberately empty, and ``relevance_grades`` for graded NDCG.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.logging import get_logger, log_evaluation
from app.evaluation.metrics import (
    aggregate_metrics,
    f1_at_k,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    reciprocal_rank,
    recall_at_k,
)
from app.ml.chunking import chunk_documents, load_documents
from app.ml.hybrid import HybridRetriever, RetrieverResult
from app.schemas.evaluation import EvalExample, PerQueryResult, RetrievalMetrics

logger = get_logger(__name__)

__all__ = [
    "DatasetNotLabelledError",
    "EvaluationDataset",
    "LabelledQuery",
    "RetrievalEvaluationRunner",
    "RetrievalEvaluationSummary",
    "load_dataset",
    "recorded_configuration",
    "score_retrieval",
]

#: ``category`` values present in the shipped dataset. Used for reporting only —
#: the runner never filters on them unless asked to.
NO_ANSWER_CATEGORY = "no_answer"

#: ``app`` is four levels below the repository root: backend/app/evaluation/.
_REPO_ROOT: Path = Path(__file__).resolve().parents[3]


def resolve_repo_path(path: str | Path) -> Path:
    """Resolve a configured path against the CWD *then* the repository root.

    ``settings.evaluation_dataset_path`` is repo-relative
    (``evaluation/datasets/retrieval_eval.jsonl``), and the same setting has to
    work from the repo root (scripts, CI) and from ``backend/`` (uvicorn, where
    the working directory is the package's parent). Trying the CWD first keeps
    an operator's explicit override working; falling back to the repo root keeps
    the default working from either place instead of failing with a path error
    that looks like a missing dataset.
    """
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    return _REPO_ROOT / candidate


class DatasetNotLabelledError(ValueError):
    """A dataset file has no query/label rows.

    Raised instead of running an evaluation that would produce an aggregate over
    zero examples, which is arithmetically fine and completely useless.
    """


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LabelledQuery:
    """One labelled row, with every field the dataset actually carried.

    ``relevant`` is an empty set for a ``no_answer`` row, which is not the same
    as a row that was mislabelled: it is a query the corpus deliberately cannot
    answer, and the runner reports how many there were rather than scoring it.
    """

    query: str
    relevant: frozenset[str]
    grades: dict[str, int] = field(default_factory=dict)
    category: str | None = None
    no_answer: bool = False
    tags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EvaluationDataset:
    """A loaded dataset plus the provenance needed to reproduce the run."""

    name: str
    path: str | None
    queries: tuple[LabelledQuery, ...]
    skipped_rows: int = 0
    format_errors: int = 0

    @property
    def num_queries(self) -> int:
        return len(self.queries)

    @property
    def num_labelled(self) -> int:
        """Rows with at least one relevant document — the metric-bearing ones."""
        return sum(1 for query in self.queries if query.relevant)

    @property
    def num_no_answer(self) -> int:
        """Rows the corpus is expected *not* to answer."""
        return sum(1 for query in self.queries if query.no_answer)

    @property
    def categories(self) -> dict[str, int]:
        """``category -> count``, including the unscored ``no_answer`` rows."""
        counts: dict[str, int] = {}
        for query in self.queries:
            if query.category:
                counts[query.category] = counts.get(query.category, 0) + 1
        return dict(sorted(counts.items()))

    def summary(self) -> dict[str, Any]:
        """Factual description of what was loaded, stored with the run."""
        return {
            "dataset_name": self.name,
            "path": self.path,
            "num_queries": self.num_queries,
            "num_labelled": self.num_labelled,
            "num_no_answer": self.num_no_answer,
            "categories": self.categories,
            "skipped_rows": self.skipped_rows,
            "format_errors": self.format_errors,
        }


def _first_present(payload: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def _parse_row(payload: dict[str, Any]) -> LabelledQuery | None:
    """Build a :class:`LabelledQuery` from one decoded JSON object.

    Returns ``None`` for a row with no query text — a line that is not a usable
    example. The caller counts those rather than pretending the dataset was
    smaller than the file.
    """
    query = _first_present(payload, "query", "question", "text")
    if not isinstance(query, str) or not query.strip():
        return None

    raw_relevant = _first_present(payload, "relevant_document_ids", "relevant_documents")
    if raw_relevant is None:
        relevant: list[str] = []
    elif isinstance(raw_relevant, list):
        relevant = [str(item) for item in raw_relevant if item is not None]
    else:
        relevant = [str(raw_relevant)]

    raw_grades = _first_present(payload, "relevance_grades", "grades")
    grades: dict[str, int] = {}
    if isinstance(raw_grades, dict):
        for doc_id, grade in raw_grades.items():
            try:
                grades[str(doc_id)] = int(grade)
            except (TypeError, ValueError):
                continue

    category = payload.get("category")
    tags = payload.get("tags")
    return LabelledQuery(
        query=query.strip(),
        relevant=frozenset(relevant),
        grades=grades,
        category=str(category) if category is not None else None,
        no_answer=bool(payload.get("no_answer", False)) or category == NO_ANSWER_CATEGORY,
        tags=tuple(str(tag) for tag in tags) if isinstance(tags, list) else (),
    )


def load_dataset(path: str | Path | None = None) -> EvaluationDataset:
    """Load ``retrieval_eval.jsonl`` (or any file of the same shape).

    Supports two shapes, both of which exist in the wild for this format:

    * **JSON Lines** — one object per line. This is what
      ``evaluation/datasets/retrieval_eval.jsonl`` is, and a line that fails to
      decode is counted and skipped rather than aborting a run over one bad row.
    * **A single object** — ``{"name": ..., "examples": [...]}``, where the
      nested rows use their own field names (``question``/``relevant_documents``).
      The ``name`` becomes the dataset name.

    Raises :class:`FileNotFoundError` when the path does not exist and
    :class:`DatasetNotLabelledError` when nothing usable came out of it.
    """
    resolved = resolve_repo_path(path or settings.evaluation_dataset_path)
    if not resolved.exists():
        raise FileNotFoundError(
            f"evaluation dataset not found: {resolved}. Build it with "
            "evaluation/scripts/build_dataset_ids.py, or pass dataset_path=..."
        )

    raw = resolved.read_text(encoding="utf-8")
    stripped = raw.lstrip()
    name = resolved.stem

    payload: Any = None
    rows: list[dict[str, Any]] = []
    if stripped.startswith("{") and "\n" not in stripped.strip():
        payload = json.loads(stripped)
        if isinstance(payload, dict) and isinstance(payload.get("examples"), list):
            name = str(payload.get("name") or name)
            rows = [row for row in payload["examples"] if isinstance(row, dict)]
        else:
            rows = [payload] if isinstance(payload, dict) else []
    else:
        format_errors = 0
        for line in raw.splitlines():
            candidate = line.strip()
            if not candidate:
                continue
            try:
                decoded = json.loads(candidate)
            except ValueError:
                format_errors += 1
                continue
            if isinstance(decoded, dict):
                # A file that is itself one object per line but wraps them all
                # still parses; ``examples`` wins over a stray top-level key.
                nested = decoded.get("examples")
                if isinstance(nested, list):
                    name = str(decoded.get("name") or name)
                    rows.extend(row for row in nested if isinstance(row, dict))
                else:
                    rows.append(decoded)
            else:
                format_errors += 1
        if format_errors:
            logger.warning(
                "evaluation.dataset.undecodable_rows",
                path=str(resolved),
                format_errors=format_errors,
            )

    if not rows:
        # One JSONL file can legitimately be a single object with a trailing
        # newline, so re-read it as a whole before giving up.
        try:
            decoded = json.loads(raw)
        except ValueError:
            # Whitespace-only, or nothing to parse at all. Either way the
            # dataset has no rows, which is the same situation as a file full
            # of undecodable lines -- the named error below covers both.
            decoded = None
        if isinstance(decoded, dict):
            nested = decoded.get("examples")
            name = str(decoded.get("name") or name)
            rows = [nested] if isinstance(nested, dict) else [
                row for row in nested or [] if isinstance(row, dict)
            ]

    parsed: list[LabelledQuery] = []
    skipped = 0
    for row in rows:
        built = _parse_row(row)
        if built is None:
            skipped += 1
            continue
        parsed.append(built)

    if not parsed:
        raise DatasetNotLabelledError(
            f"{resolved} contained {len(rows)} rows but no usable labelled query. "
            "Each row needs a 'query' string and a relevant-document list."
        )

    return EvaluationDataset(
        name=name,
        path=str(resolved),
        queries=tuple(parsed),
        skipped_rows=skipped,
        format_errors=0,
    )


def to_eval_examples(dataset: EvaluationDataset) -> list[EvalExample]:
    """Convert to the API's :class:`EvalExample` shape for request echoing.

    ``no_answer`` rows cannot be expressed — the schema requires at least one
    relevant document — so they are excluded here. They are still scored by the
    runner's own accounting, which is why this is a projection and not the
    loader's internal type.
    """
    return [
        EvalExample(
            query=query.query,
            relevant_documents=sorted(query.relevant),
            relevance_grades=query.grades or None,
            tags=list(query.tags),
        )
        for query in dataset.queries
        if query.relevant
    ]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ScoredQuery:
    """Per-query metrics plus the ids they were computed from.

    Holds both id lists on purpose: a stored result whose relevant ids are
    missing cannot be re-checked, and a recall of 0.4 with no record of what was
    relevant is not a measurement anyone can investigate.
    """

    result: PerQueryResult

    def as_metrics(self) -> dict[str, float]:
        """Flat metric dict for :func:`app.evaluation.metrics.aggregate_metrics`."""
        return {
            "precision": self.result.precision,
            "recall": self.result.recall,
            "f1": self.result.f1,
            "reciprocal_rank": self.result.reciprocal_rank,
            "ndcg": self.result.ndcg,
            "hit": 1.0 if self.result.hit else 0.0,
            "num_retrieved": float(len(self.result.retrieved_document_ids)),
            "num_zero_results": 1.0 if not self.result.retrieved_document_ids else 0.0,
        }


def score_retrieval(
    retrieved: Sequence[str],
    relevant: Iterable[str],
    k: int,
    *,
    grades: dict[str, int] | None = None,
) -> ScoredQuery:
    """Score one query's ranking with :mod:`app.evaluation.metrics`.

    ``retrieved`` is the retriever's ranked id list; ``k`` slices it. Every
    function here is imported from the metrics module rather than reimplemented,
    which is the only way to guarantee the runner and the unit tests agree.
    """
    ranked = list(retrieved)
    relevant_set = set(relevant)
    retrieved_top_k = ranked[:k]

    hit = hit_rate_at_k(ranked, relevant_set, k)
    return ScoredQuery(
        result=PerQueryResult(
            # Truncated so a stored row does not carry 4x the ids at k=5; the
            # run's config records the k these were computed at.
            query="",
            k=k,
            precision=precision_at_k(ranked, relevant_set, k),
            recall=recall_at_k(ranked, relevant_set, k),
            f1=f1_at_k(ranked, relevant_set, k),
            reciprocal_rank=reciprocal_rank(ranked, relevant_set),
            ndcg=ndcg_at_k(ranked, relevant_set, k, grades=grades),
            hit=hit,
            retrieved_document_ids=retrieved_top_k,
            relevant_document_ids=sorted(relevant_set),
            missed_document_ids=sorted(relevant_set - set(retrieved_top_k)),
        )
    )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def build_retrieval_metrics(
    scored: Sequence[ScoredQuery],
    k: int,
    *,
    num_queries: int,
    no_answer_queries: int = 0,
) -> RetrievalMetrics:
    """Aggregate scored queries into the API's per-K metric row.

    Only *labelled* queries feed these means. ``num_queries`` is the labelled
    count, and ``zero_result_rate`` is a mean over the same set, so every field
    describes the same population.
    """
    if not scored:
        raise ValueError("cannot aggregate an empty result set")

    rows = [item.result for item in scored]
    return RetrievalMetrics(
        k=k,
        precision_at_k=_mean([row.precision for row in rows]),
        recall_at_k=_mean([row.recall for row in rows]),
        f1_at_k=_mean([row.f1 for row in rows]),
        mrr=_mean([row.reciprocal_rank for row in rows]),
        ndcg_at_k=_mean([row.ndcg for row in rows]),
        hit_rate_at_k=_mean([1.0 if row.hit else 0.0 for row in rows]),
        zero_result_rate=_mean(
            [1.0 if not row.retrieved_document_ids else 0.0 for row in rows]
        ),
        num_queries=num_queries,
        avg_documents_retrieved=_mean([float(len(row.retrieved_document_ids)) for row in rows]),
    )


# ---------------------------------------------------------------------------
# Configuration record
# ---------------------------------------------------------------------------


def recorded_configuration(
    retriever: HybridRetriever,
    *,
    ks: Sequence[int],
    dataset: EvaluationDataset,
    num_indexed_chunks: int,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The retrieval configuration a run used, read from the retriever itself.

    Taken from ``retriever.configuration()`` rather than re-read from
    ``settings``: the retriever holds the values actually in force (including any
    injected components), and a config reconstructed from global settings can
    describe a run that never happened.

    Overrides are recorded under ``overrides`` rather than merged over the top,
    so a comparison can tell "this setting was changed for the run" from "this
    is what the setting was".
    """
    config: dict[str, Any] = {
        **retriever.configuration(),
        "k_values": list(ks),
        "primary_k": ks[0] if ks else None,
        "num_indexed_chunks": num_indexed_chunks,
        "dataset": dataset.summary(),
    }
    if overrides:
        config["overrides"] = dict(overrides)
    return config


# ---------------------------------------------------------------------------
# Run summary
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationSummary:
    """Aggregate outcome of one evaluation run."""

    run_id: str | None
    name: str
    dataset_name: str
    k: int
    metrics: dict[str, float]
    metrics_by_k: dict[int, RetrievalMetrics]
    per_query: list[PerQueryResult]
    configuration: dict[str, Any]
    num_labelled: int
    num_no_answer: int
    num_unresolved: int
    duration_ms: float
    persisted: bool
    no_answer_results: dict[str, Any] = field(default_factory=dict)

    @property
    def num_queries(self) -> int:
        return self.num_labelled

    def as_metric_rows(self) -> dict[str, float]:
        """``{"k=5": {...}}`` — the shape persisted into ``EvaluationRun.metrics``.

        Keyed by k rather than nested, because the regression detector deltas a
        flat mapping of metric names and two runs at different k values must not
        collide on the same key.
        """
        return {f"k={k}": values.model_dump() for k, values in sorted(self.metrics_by_k.items())}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class RetrievalEvaluationRunner:
    """Runs a labelled dataset through the real retriever and records the run.

    Construct it with an already-indexed retriever, or let it build the index
    from ``settings.knowledge_base_path`` on first use. The index build is the
    expensive part of the whole evaluation (an embedding checkpoint plus a FAISS
    write), so it is a constructor concern rather than something repeated per
    query.

    Persistence is optional and off the critical path: with ``persist=True`` the
    runner writes one ``EvaluationRun`` plus one ``EvaluationResult`` per scored
    query and returns a summary carrying the new run id. It never rolls the
    metrics back on a write failure — the measurements already happened and are
    still returned, with the failure logged.
    """

    def __init__(
        self,
        *,
        retriever: HybridRetriever | None = None,
        knowledge_base_path: str | Path | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        persist: bool = True,
    ) -> None:
        self._retriever = retriever if retriever is not None else HybridRetriever()
        self._knowledge_base_path = resolve_repo_path(
            knowledge_base_path or settings.knowledge_base_path
        )
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._persist = persist

    # ------------------------------------------------------------------
    # Corpus
    # ------------------------------------------------------------------

    def build_index(self, *, force: bool = False) -> int:
        """Chunk and index the knowledge base; returns the chunk count.

        A no-op when the retriever is already indexed unless ``force``, because
        re-embedding a few hundred chunks is seconds of work that a second
        caller should not pay for by default.
        """
        if self._retriever.is_indexed and not force:
            logger.info(
                "evaluation.index.reused",
                chunks=self._retriever.size,
            )
            return self._retriever.size

        documents = load_documents(self._knowledge_base_path)
        chunks = chunk_documents(documents, self._chunk_size, self._chunk_overlap)
        return self._retriever.index_documents(chunks)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def run(
        self,
        *,
        session: AsyncSession | None = None,
        dataset: EvaluationDataset | None = None,
        dataset_path: str | Path | None = None,
        ks: Sequence[int] | None = None,
        name: str | None = None,
        application_id: Any | None = None,
        config_override: dict[str, Any] | None = None,
        persist: bool | None = None,
        limit: int | None = None,
    ) -> RetrievalEvaluationSummary:
        """Evaluate every labelled query at every k in ``ks``.

        ``ks`` is sorted and de-duplicated; the first value is the primary k,
        matching ``RetrievalEvaluationRequest.k`` being the headline number and
        the rest additional reporting points.

        The ``no_answer`` rows are retrieved but *not* scored against an empty
        relevance set — ``precision_at_k`` and friends return 0.0 for those by
        design, and averaging them in would drag every metric down by
        construction rather than because retrieval is worse. They are counted and
        their raw retrieved ids are reported, which is what makes the dataset's
        deliberate negatives auditable instead of invisible.
        """
        loaded = dataset if dataset is not None else load_dataset(dataset_path)
        k_values = sorted({int(value) for value in (ks or [settings.default_top_k]) if value > 0})
        if not k_values:
            raise ValueError("at least one positive k is required")

        queries = list(loaded.queries)
        if limit is not None:
            queries = queries[: max(0, int(limit))]

        started_wall = datetime.now(timezone.utc)
        started = time.perf_counter()

        num_indexed = self.build_index()
        if not self._retriever.is_indexed:
            # Refuse rather than report a confident zero. Every metric would be
            # 0.0 for a reason that has nothing to do with retrieval quality, and
            # that number would be indistinguishable from a real failure.
            raise RuntimeError(
                f"retriever is not indexed after building {num_indexed} chunks from "
                f"{self._knowledge_base_path}; refusing to report metrics against an "
                "empty index"
            )

        configuration = recorded_configuration(
            self._retriever,
            ks=k_values,
            dataset=loaded,
            num_indexed_chunks=num_indexed,
            overrides=config_override,
        )
        # The largest k actually requested is what the retriever has to return:
        # scoring at k=10 from a k=5 result would report a truncation, not a
        # recall.
        fetch_k = max(k_values)

        labelled = [query for query in queries if query.relevant]
        no_answer = [query for query in queries if query.no_answer]

        scored_by_k: dict[int, list[ScoredQuery]] = {k: [] for k in k_values}
        per_query: list[PerQueryResult] = []
        timings: list[float] = []
        num_unresolved = 0

        for query in labelled:
            result = self._retriever.run(query.query, top_k=fetch_k, with_timings=True)
            timings.append(float(result.timings_ms.get("total_ms") or 0.0))
            ranked = result.document_ids()
            if not ranked:
                num_unresolved += 1

            for k in k_values:
                scored = score_retrieval(
                    ranked, query.relevant, k, grades=query.grades or None
                )
                scored.result.query = query.query
                scored_by_k[k].append(scored)

            # One row per query at the primary k: that is what a reader
            # comparing two runs actually looks at.
            primary = scored_by_k[k_values[0]][-1]
            per_query.append(primary.result)

        no_answer_results = self._run_no_answer(no_answer, fetch_k)

        metrics_by_k = {
            k: build_retrieval_metrics(
                scored_by_k[k], k, num_queries=len(scored_by_k[k])
            )
            for k in k_values
        }
        primary_metrics = metrics_by_k[k_values[0]]
        metrics: dict[str, float] = dict(aggregate_metrics(
            [item.as_metrics() for item in scored_by_k[k_values[0]]], k_values[0]
        ))
        # The schema's own fields are copied in verbatim so the stored aggregate
        # is directly comparable with a freshly computed one.
        metrics.update(primary_metrics.model_dump())
        metrics["zero_result_rate"] = primary_metrics.zero_result_rate
        metrics["num_queries"] = float(len(labelled))

        duration_ms = (time.perf_counter() - started) * 1000.0
        run_name = name or f"{loaded.name}@{started_wall.isoformat(timespec='seconds')}"

        configuration["measured"] = {
            "avg_retrieval_ms": _mean(timings),
            "max_retrieval_ms": max(timings) if timings else 0.0,
            "num_labelled_queries": len(labelled),
            "num_no_answer_queries": len(no_answer),
            "num_queries_with_no_results": num_unresolved,
            "fetch_k": fetch_k,
            "started_at": started_wall.isoformat(),
        }

        should_persist = self._persist if persist is None else bool(persist)
        run_id: str | None = None
        if should_persist:
            if session is None:
                logger.warning(
                    "evaluation.persistence.skipped_no_session",
                    dataset=loaded.name,
                )
            else:
                run_id = await self._persist_run(
                    session=session,
                    name=run_name,
                    dataset_name=loaded.name,
                    application_id=application_id,
                    k=k_values[0],
                    num_queries=len(labelled),
                    metrics=metrics,
                    metrics_by_k=metrics_by_k,
                    configuration=configuration,
                    per_query=per_query,
                    duration_ms=duration_ms,
                    started_at=started_wall,
                )

        log_evaluation(
            run_id or run_name,
            "retrieval",
            duration_ms,
            len(labelled),
        )
        # `loaded.summary()` already carries dataset_name / num_queries /
        # num_labelled. Merge it into one dict rather than spreading it
        # alongside explicit keywords: `**d, k=v` is itself a duplicate-kwarg
        # TypeError in plain Python (raised before structlog is reached), and
        # it fired on every completed retrieval run. The explicit values below
        # are the run-scoped ones, so they deliberately win.
        event_fields = {
            **loaded.summary(),
            "run_id": run_id,
            "dataset": loaded.name,
            "k": k_values,
            "num_queries": len(labelled),
            "mrr": round(primary_metrics.mrr, 4),
            "ndcg": round(primary_metrics.ndcg_at_k, 4),
            "duration_ms": round(duration_ms, 2),
        }
        logger.info("evaluation.retrieval.completed", **event_fields)

        return RetrievalEvaluationSummary(
            run_id=run_id,
            name=run_name,
            dataset_name=loaded.name,
            k=k_values[0],
            metrics=metrics,
            metrics_by_k=metrics_by_k,
            per_query=per_query,
            configuration=configuration,
            num_labelled=len(labelled),
            num_no_answer=len(no_answer),
            num_unresolved=num_unresolved,
            duration_ms=duration_ms,
            persisted=run_id is not None,
            no_answer_results=no_answer_results,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run_no_answer(
        self, queries: Sequence[LabelledQuery], fetch_k: int
    ) -> dict[str, Any]:
        """Retrieve the out-of-scope queries and report what came back.

        Deliberately not scored. These rows exist so a corpus can be checked for
        whether it answers questions it should not; the useful number is how many
        of them returned *anything* at all, which is a property of the corpus,
        not of ranking quality.
        """
        if not queries:
            return {"count": 0}

        rows: list[dict[str, Any]] = []
        returned_anything = 0
        for query in queries:
            result: RetrieverResult = self._retriever.run(
                query.query, top_k=fetch_k, with_timings=False
            )
            ids = result.document_ids()
            returned_anything += 1 if ids else 0
            rows.append(
                {
                    "query": query.query,
                    "num_retrieved": len(ids),
                    "retrieved_document_ids": ids[:fetch_k],
                    "top_score": result.top_score,
                }
            )
        return {
            "count": len(queries),
            "num_returned_results": returned_anything,
            "queries": rows,
        }

    async def _persist_run(
        self,
        *,
        session: AsyncSession,
        name: str,
        dataset_name: str,
        application_id: Any,
        k: int,
        num_queries: int,
        metrics: dict[str, float],
        metrics_by_k: dict[int, RetrievalMetrics],
        configuration: dict[str, Any],
        per_query: Sequence[PerQueryResult],
        duration_ms: float,
        started_at: datetime,
    ) -> str | None:
        """Write the ``EvaluationRun`` and its per-query rows.

        The run row is inserted and flushed first so the results have a parent
        to reference. Stored metrics are the k-keyed rows, which keeps two runs
        at different k values from colliding when they are diffed.
        """
        from app.models import EvaluationResult, EvaluationRun, RunStatus, utcnow

        run = EvaluationRun(
            name=name,
            application_id=application_id,
            dataset_name=dataset_name,
            evaluation_type="retrieval",
            status=RunStatus.COMPLETED,
            k=k,
            num_queries=num_queries,
            metrics=self._stored_metrics(metrics, metrics_by_k),
            config=configuration,
            started_at=started_at,
            completed_at=utcnow(),
            duration_ms=round(duration_ms, 3),
        )
        session.add(run)
        await session.flush()

        for item in per_query:
            session.add(
                EvaluationResult(
                    run_id=run.id,
                    query=item.query,
                    k=item.k,
                    metrics={
                        "precision": item.precision,
                        "recall": item.recall,
                        "f1": item.f1,
                        "reciprocal_rank": item.reciprocal_rank,
                        "ndcg": item.ndcg,
                        "hit": item.hit,
                    },
                    retrieved_document_ids=item.retrieved_document_ids,
                    relevant_document_ids=item.relevant_document_ids,
                )
            )

        try:
            await session.commit()
        except Exception:
            await session.rollback()
            # The measurements are real and already returned by the caller; a
            # failed write must not discard them or crash an evaluation.
            logger.exception("evaluation.retrieval.persist_failed", dataset=dataset_name)
            return None

        logger.info("evaluation.retrieval.persisted", run_id=str(run.id), rows=len(per_query))
        return str(run.id)

    @staticmethod
    def _stored_metrics(
        metrics: dict[str, float], metrics_by_k: dict[int, RetrievalMetrics]
    ) -> dict[str, Any]:
        """The aggregate blob stored on the run.

        Primary-k scalars sit at the top level for the regression detector, and
        the full per-k table sits under ``by_k`` so a k=10 comparison does not
        overwrite a k=5 one.
        """
        return {
            **metrics,
            "by_k": {str(k): value.model_dump() for k, value in sorted(metrics_by_k.items())},
        }
