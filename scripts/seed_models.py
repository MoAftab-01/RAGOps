"""Seed the reference ``models`` rows used by cost analytics.

Run from the repository root:

    .venv/Scripts/python.exe scripts/seed_models.py

Every row is derived from :data:`app.utils.pricing.DEFAULT_PRICING` -- the same
table ``calculate_cost`` reads from -- so a seeded model can never disagree with
what the cost calculator will later charge for it. Nothing here is invented at
call time: the local models really do cost 0.0, and the hosted entries carry
the simulated list prices that every UI surface labels as simulated.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

from sqlalchemy import select  # noqa: E402

from app.core.database import session_scope  # noqa: E402
from app.models import Model  # noqa: E402
from app.utils.pricing import DEFAULT_PRICING  # noqa: E402


async def seed() -> int:
    """Insert any missing models. Existing rows are left untouched."""
    created = 0

    async with session_scope() as session:
        for pricing in DEFAULT_PRICING:
            existing = await session.scalar(
                select(Model).where(
                    Model.provider == pricing.provider,
                    Model.name == pricing.model_name,
                )
            )
            if existing is not None:
                continue

            session.add(
                Model(
                    name=pricing.model_name,
                    provider=pricing.provider,
                    context_window=pricing.context_window,
                    is_local=pricing.is_local,
                    input_cost_per_1k=pricing.input_cost_per_1k,
                    output_cost_per_1k=pricing.output_cost_per_1k,
                    description=pricing.description,
                )
            )
            created += 1

        await session.commit()

    return created


async def _main() -> int:
    created = await seed()
    print(f"Seeded {created} model(s).")
    print("Note: hosted-model prices are SIMULATED list prices; local models cost 0.00.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
