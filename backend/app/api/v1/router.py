"""The v1 router: one place that knows the whole HTTP surface.

Order is documentation order, not dispatch order — FastAPI matches by path, so
the order here decides which routes appear first in ``/docs``. It follows the API
contract top to bottom, so the generated documentation reads the way the contract
does.

Two routers share the ``/evaluations`` prefix and two share ``/analytics``: this
is deliberate, because the contract groups them that way. They are registered in
the order the contract lists their endpoints so the OpenAPI paths do not appear
shuffled, and because they are separate modules with separate concerns — retrieval
evaluation and answer evaluation are different pipelines that happen to be filed
under one heading.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1 import (
    anomalies,
    answer_eval,
    applications,
    collector,
    cost_quality,
    dashboard,
    experiments,
    health,
    models,
    optimization,
    organizations,
    rag,
    retrieval,
    token_analytics,
    traces,
)

api_router = APIRouter()

# Health first: it is what a load balancer and an operator hit, and it is the
# only endpoint that answers when the rest of the system is down.
api_router.include_router(health.router)

# Analytics, then the raw records they are computed from.
api_router.include_router(dashboard.router)
api_router.include_router(token_analytics.router)
api_router.include_router(cost_quality.router)
api_router.include_router(traces.router)

# Evaluation: retrieval before answer, as the contract lists them.
api_router.include_router(retrieval.router)
api_router.include_router(answer_eval.router)

# Analysis of recorded data, then the configuration it was recorded under.
api_router.include_router(anomalies.router)
api_router.include_router(optimization.router)
api_router.include_router(experiments.router)
api_router.include_router(applications.router)
api_router.include_router(models.router)

# The company layer itself. Platform-privileged and last of the configuration
# group: it is where organizations and their credentials are created, so it
# reads as the administrative surface rather than part of the instrumentation.
api_router.include_router(organizations.router)

# The demo and the telemetry pipeline itself.
api_router.include_router(rag.router)
api_router.include_router(collector.router)

__all__ = ["api_router"]
