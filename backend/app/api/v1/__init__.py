"""Version 1 of the RAGOps HTTP API.

The routers are imported here rather than in :mod:`app.api.v1.router` so that a
single import pulls the whole surface into memory, which is what makes a missing
module fail at start-up instead of on the first request that reaches it.
"""

from __future__ import annotations

from app.api.v1.router import api_router

__all__ = ["api_router"]
