"""ORM model registry.

Importing this package registers every table on ``Base.metadata``, which is
what Alembic autogeneration and ``create_all`` rely on.
"""

from app.models.analytics import (
    Anomaly,
    AnomalyType,
    AnswerEvaluation,
    Document,
    EvaluationResult,
    EvaluationRun,
    Experiment,
    ExperimentVariant,
    OptimizationRecommendation,
    RunStatus,
    Severity,
)
from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin, utcnow
from app.models.core import (
    Application,
    LLMCall,
    Model,
    RetrievedDocument,
    RetrievalCall,
    Span,
    SpanKind,
    Trace,
    TraceKind,
    TraceStatus,
    User,
)
from app.models.tenancy import (
    DEFAULT_KEY_SCOPES,
    KEY_PREFIX_LENGTH,
    ApiKey,
    Organization,
)

__all__ = [
    "DEFAULT_KEY_SCOPES",
    "KEY_PREFIX_LENGTH",
    "Anomaly",
    "AnomalyType",
    "AnswerEvaluation",
    "ApiKey",
    "Application",
    "Base",
    "Document",
    "EvaluationResult",
    "EvaluationRun",
    "Experiment",
    "ExperimentVariant",
    "LLMCall",
    "Model",
    "OptimizationRecommendation",
    "Organization",
    "RetrievedDocument",
    "RetrievalCall",
    "RunStatus",
    "Severity",
    "Span",
    "SpanKind",
    "TimestampMixin",
    "Trace",
    "TraceKind",
    "TraceStatus",
    "UUIDPrimaryKeyMixin",
    "User",
    "utcnow",
]
