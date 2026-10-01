"""Evaluation, anomaly, recommendation and experiment schemas."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.schemas.common import ORMModel


# ---------------------------------------------------------------------------
# Evaluation datasets and runs
# ---------------------------------------------------------------------------


class EvalExample(BaseModel):
    """One labelled query in a retrieval evaluation dataset.

    ``relevant_documents`` holds document ids as they appear in the knowledge
    base (``Document.external_id``), so a dataset stays valid across re-ingestion.
    """

    query: str = Field(min_length=1, max_length=4000)
    relevant_documents: list[str] = Field(min_length=1, max_length=100)
    # Optional graded relevance for NDCG: id -> gain (2 = highly relevant).
    relevance_grades: dict[str, int] | None = None
    reference_answer: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalEvaluationRequest(BaseModel):
    """Run retrieval metrics over a labelled dataset.

    Either point at an existing ``application`` (which runs the real retriever)
    or supply ``retrieved_document_ids`` per example to score a result set that
    was produced elsewhere. The second form is what evaluates a production
    trace log without re-running retrieval.
    """

    name: str | None = Field(default=None, max_length=255)
    application: str | None = Field(default=None, max_length=128)
    dataset_name: str = Field(default="retrieval_eval", max_length=255)
    k: int = Field(default=5, ge=1, le=50)
    ks: list[int] = Field(default_factory=list, description="Additional K values to report")
    examples: list[EvalExample] | None = None
    # When true, the server loads examples from the dataset path instead.
    load_from_path: bool = True
    # Optional override of the retrieval configuration under test.
    config_override: dict[str, Any] = Field(default_factory=dict)
    persist: bool = True

    @property
    def all_k(self) -> list[int]:
        values = {self.k, *self.ks}
        return sorted(values)


class RetrievalMetrics(BaseModel):
    """Aggregated retrieval metrics for one K."""

    k: int
    precision_at_k: float
    recall_at_k: float
    f1_at_k: float
    mrr: float
    ndcg_at_k: float
    hit_rate_at_k: float
    # How many examples retrieved nothing at all — a distinct failure mode from
    # retrieving the wrong things.
    zero_result_rate: float
    num_queries: int
    avg_documents_retrieved: float


class PerQueryResult(BaseModel):
    query: str
    k: int
    precision: float
    recall: float
    f1: float
    reciprocal_rank: float
    ndcg: float
    hit: bool
    retrieved_document_ids: list[str]
    relevant_document_ids: list[str]
    # Per-query breakdown of which relevant docs were missed, for debugging.
    missed_document_ids: list[str] = Field(default_factory=list)
    latency_ms: float | None = None


class EvaluationRunSummary(ORMModel):
    # Nullable only for an unpersisted run. A run submitted with
    # ``persist: false`` is executed and measured but writes no row, so there is
    # no id to hand back -- and inventing one would produce a link on the
    # evaluation page that 404s when followed. ``status`` already distinguishes
    # the two cases ("completed" vs "not_persisted"), so a null id never has to
    # be guessed at by the client.
    id: uuid.UUID | None = None
    name: str
    application_id: uuid.UUID | None
    dataset_name: str
    evaluation_type: str
    status: str
    k: int
    num_queries: int
    metrics: dict[str, Any] | None
    config: dict[str, Any] | None
    started_at: datetime | None
    completed_at: datetime | None
    duration_ms: float | None
    notes: str | None
    error: str | None
    created_at: datetime


class EvaluationRunDetail(EvaluationRunSummary):
    results: list[PerQueryResult] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Answer evaluation
# ---------------------------------------------------------------------------


class AnswerEvaluationItem(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    answer: str = Field(min_length=1, max_length=100_000)
    context: list[str] = Field(default_factory=list)
    reference_answer: str | None = None
    trace_id: str | None = None
    relevant_documents: list[str] = Field(default_factory=list)


class AnswerEvaluationRequest(BaseModel):
    name: str | None = Field(default=None, max_length=255)
    application: str | None = Field(default=None, max_length=128)
    items: list[AnswerEvaluationItem] = Field(min_length=1, max_length=500)
    # Off by default: deterministic metrics need no model.
    use_llm_judge: bool = False
    judge_model: str | None = None
    persist: bool = True


class ClaimEvaluation(BaseModel):
    """One extracted claim and whether the context supports it."""

    claim: str
    supported: bool
    best_matching_context_index: int | None
    similarity: float
    method: str = Field(description="deterministic_alignment | embedding_similarity")


class AnswerEvaluationResult(BaseModel):
    question: str
    faithfulness: float
    context_relevance: float
    answer_relevance: float
    citation_coverage: float
    unsupported_claim_ratio: float
    judge_faithfulness: float | None = None
    judge_answer_relevance: float | None = None
    judge_model: str | None = None
    method: str = Field(
        description="Which family of metrics produced these numbers"
    )
    claims: list[ClaimEvaluation] = Field(default_factory=list)
    duration_ms: float | None = None


class AnswerEvaluationSummary(BaseModel):
    """Aggregate of an answer evaluation. Deterministic and judge columns are
    kept apart so the UI can never present a judge number as a measured one."""

    num_items: int
    method: str
    faithfulness: float
    context_relevance: float
    answer_relevance: float
    citation_coverage: float
    unsupported_claim_ratio: float
    judge_faithfulness: float | None = None
    judge_answer_relevance: float | None = None
    judge_model: str | None = None
    per_item: list[AnswerEvaluationResult] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Regression detection
# ---------------------------------------------------------------------------


class MetricDelta(BaseModel):
    metric: str
    baseline: float
    candidate: float
    absolute_change: float
    relative_change_pct: float
    is_regression: bool
    direction: str = Field(description="higher_is_better | lower_is_better | neutral")


class ConfigDifference(BaseModel):
    """A recorded, factual difference between two runs' configurations.

    Only keys that actually changed appear here — RAGOps never speculates
    about causes beyond what the two configs literally say.
    """

    key: str
    baseline_value: Any
    candidate_value: Any


class RegressionReport(BaseModel):
    baseline_run_id: uuid.UUID
    candidate_run_id: uuid.UUID
    baseline_name: str
    candidate_name: str
    deltas: list[MetricDelta] = Field(default_factory=list)
    config_differences: list[ConfigDifference] = Field(default_factory=list)
    has_regression: bool
    regression_summary: str | None = None
    # Phrased as "these recorded settings changed" — deliberately not "cause".
    contributing_config_changes: list[str] = Field(default_factory=list)
