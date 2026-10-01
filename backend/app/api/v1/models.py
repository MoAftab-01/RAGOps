"""The LLM registry: which models exist, and what they are assumed to cost.

The registry is what makes every cost figure elsewhere in RAGOps interpretable,
so it is a first-class read/write surface rather than a config file. The prices it
holds are a *simulated* list price used to compare configurations against metered
providers — local inference really does cost nothing. ``is_local`` is the flag
that says which of those two a row is, and
:func:`~app.utils.pricing.pricing_label` is what turns it into the sentence shown
beside the number.
"""

from __future__ import annotations

from fastapi import APIRouter
from sqlalchemy import select

from app.api.deps import DbSession, WriteGuard
from app.core.cache import invalidate
from app.core.logging import get_logger
from app.models import Model
from app.schemas.insights import ModelOut, ModelUpsert

logger = get_logger(__name__)

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=list[ModelOut], summary="List registered models")
async def list_models(session: DbSession) -> list[ModelOut]:
    """Every registered model, provider then name.

    Unordered by cost or popularity on purpose: this is a registry, and a sorted
    registry would imply a ranking the data does not contain. A model absent
    from this list is not "free" — it is *unpriced*, and every cost figure that
    involves it is reported with that uncertainty rather than as zero.
    """
    stmt = select(Model).order_by(Model.provider.asc(), Model.name.asc())
    rows = (await session.execute(stmt)).scalars().all()
    return [ModelOut.model_validate(row) for row in rows]


@router.put(
    "",
    response_model=ModelOut,
    summary="Register or update a model",
    dependencies=[WriteGuard],
)
async def upsert_model(session: DbSession, payload: ModelUpsert) -> ModelOut:
    """Insert or update the registry entry for ``(provider, name)``.

    Upsert rather than create-or-409 because the identity of a model is that
    pair — the same model is legitimately re-registered as its context window
    or price is corrected, and a client that has to first GET to learn whether
    it is creating or updating will always be wrong once.
    """
    stmt = select(Model).where(Model.provider == payload.provider, Model.name == payload.name)
    model = (await session.execute(stmt)).scalar_one_or_none()

    if model is None:
        model = Model(provider=payload.provider, name=payload.name)
        session.add(model)
        action = "created"
    else:
        action = "updated"

    # Assigning every field, including the ones defaulting to zero, so an
    # explicit 0.0 from the caller is recorded as 0.0 rather than silently
    # leaving the previous price in place.
    model.context_window = payload.context_window
    model.is_local = payload.is_local
    model.input_cost_per_1k = payload.input_cost_per_1k
    model.output_cost_per_1k = payload.output_cost_per_1k
    model.description = payload.description
    await session.flush()

    # Cost analytics resolve the registry per request, so a price change has to
    # invalidate them or the old figures keep being served for the cache TTL.
    await invalidate("analytics")
    logger.info(
        "models.upserted",
        provider=payload.provider,
        model=payload.name,
        action=action,
        is_local=payload.is_local,
    )
    return ModelOut.model_validate(model)
