"""Redis client and a small JSON cache helper.

Analytics queries are read-mostly and identical between refreshes, so they are
memoised briefly. The cache is strictly an optimisation: every helper degrades
to calling the underlying function when Redis is unavailable, which keeps the
app runnable without the service.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

import redis.asyncio as aioredis
from redis.exceptions import RedisError

from app.config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T")

_redis: aioredis.Redis | None = None


def get_redis() -> aioredis.Redis | None:
    """Return the shared Redis client, or ``None`` if it cannot be created."""
    global _redis
    if _redis is None:
        try:
            _redis = aioredis.from_url(
                settings.redis_url, encoding="utf-8", decode_responses=True
            )
        except Exception as exc:  # pragma: no cover - construction rarely fails
            logger.warning("redis_client_init_failed", extra={"error": str(exc)})
            return None
    return _redis


async def close_redis() -> None:
    global _redis
    if _redis is not None:
        await _redis.aclose()
        _redis = None


async def ping() -> bool:
    client = get_redis()
    if client is None:
        return False
    try:
        return bool(await client.ping())
    except RedisError:
        return False


def make_key(namespace: str, **parts: Any) -> str:
    """Build a stable cache key from keyword parts.

    A digest of the repr keeps long filter combinations from producing keys
    with unbalanced braces (which Redis rejects).
    """
    payload = json.dumps(parts, sort_keys=True, default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    return f"ragops:{namespace}:{digest}"


async def cached(
    namespace: str,
    ttl: int | None = None,
    **parts: Any,
) -> Any | None:
    """Read a cached value, or ``None`` on miss / Redis outage."""
    if not settings.analytics_cache_enabled:
        return None
    client = get_redis()
    if client is None:
        return None
    try:
        raw = await client.get(make_key(namespace, **parts))
    except RedisError as exc:
        logger.warning("cache_read_failed", extra={"namespace": namespace, "error": str(exc)})
        return None
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


async def set_cached(
    namespace: str,
    value: Any,
    ttl: int | None = None,
    **parts: Any,
) -> None:
    """Store a JSON-serialisable value. Failures are logged, never raised."""
    if not settings.analytics_cache_enabled:
        return
    client = get_redis()
    if client is None:
        return
    key = make_key(namespace, **parts)
    effective_ttl = ttl if ttl is not None else settings.analytics_cache_ttl_seconds
    try:
        await client.set(
            key, json.dumps(value, default=str), ex=max(1, effective_ttl)
        )
    except (RedisError, TypeError, ValueError) as exc:
        logger.warning(
            "cache_write_failed", extra={"namespace": namespace, "error": str(exc)}
        )


async def invalidate(namespace: str | None = None) -> int:
    """Drop cached analytics. Called whenever new telemetry lands."""
    client = get_redis()
    if client is None:
        return 0
    try:
        if namespace is None:
            keys = [k async for k in client.scan_iter(match="ragops:analytics:*")]
            keys += [k async for k in client.scan_iter(match="ragops:evaluation:*")]
        else:
            keys = [k async for k in client.scan_iter(match=f"ragops:{namespace}:*")]
        if keys:
            await client.delete(*keys)
        return len(keys)
    except RedisError as exc:
        logger.warning("cache_invalidate_failed", extra={"error": str(exc)})
        return 0


async def cached_call(
    namespace: str,
    factory: Callable[[], Awaitable[T]],
    ttl: int | None = None,
    **parts: Any,
) -> T:
    """Memoise an async factory. This is the helper analytics services use."""
    hit = await cached(namespace, ttl=ttl, **parts)
    if hit is not None:
        return hit  # type: ignore[return-value]
    value = await factory()
    await set_cached(namespace, value, ttl=ttl, **parts)
    return value
