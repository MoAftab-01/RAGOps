"""Structured logging for the SDK, decoupled from the RAGOps backend.

The server side uses ``app.core.logging.get_logger`` (structlog). An SDK, by
definition, runs in someone else's process and cannot import the server package,
so it ships a tiny, self-contained equivalent with the same call shape
(``get_logger(__name__).info(event, **keyvals)``).

``structlog`` is used when it happens to be installed so SDK logs interleave
cleanly with an application's own structlog output; otherwise it degrades to
key=value lines on the stdlib root logger. The SDK declares only ``httpx`` and
``pydantic`` as dependencies, so structlog must never be a hard requirement.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

_LOGGER_NAME = "ragops.sdk"
_configured = False


class _KeyValueFormatter(logging.Formatter):
    """Render ``event`` plus keyword context as a single ``key=value`` line.

    Kept deliberately small: an SDK should not impose a logging format on the
    host application, only emit lines that are readable on their own.
    """

    def format(self, record: logging.LogRecord) -> str:
        parts: list[str] = [record.getMessage()]
        for key, value in getattr(record, "extra_fields", {}).items():
            parts.append(f"{key}={value!r}")
        return " ".join(parts)


def _configure() -> None:
    global _configured
    if _configured:
        return
    _configured = True

    root = logging.getLogger(_LOGGER_NAME)
    if not root.handlers:
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setFormatter(_KeyValueFormatter())
        root.addHandler(handler)
    # The SDK is an observer: it must stay quiet unless the host opts in.
    root.setLevel(logging.WARNING)
    root.propagate = False


class _SDKLogger:
    """Minimal structlog-compatible logger (``info``/``warning``/``error``/...)."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._logger = logging.getLogger(f"{_LOGGER_NAME}.{name}")

    def _log(self, level: int, event: str, **kwargs: Any) -> None:
        # structlog's signature is (event, **keyvals); we keep the same shape so
        # a swap to real structlog elsewhere changes nothing at the call sites.
        self._logger.log(level, event, extra={"extra_fields": kwargs})

    def debug(self, event: str, **kwargs: Any) -> None:
        self._log(logging.DEBUG, event, **kwargs)

    def info(self, event: str, **kwargs: Any) -> None:
        self._log(logging.INFO, event, **kwargs)

    def warning(self, event: str, **kwargs: Any) -> None:
        self._log(logging.WARNING, event, **kwargs)

    def error(self, event: str, **kwargs: Any) -> None:
        self._log(logging.ERROR, event, **kwargs)

    def exception(self, event: str, **kwargs: Any) -> None:
        self._logger.exception(event, extra={"extra_fields": kwargs})


def get_logger(name: str | None = None) -> Any:
    """Return a structured logger. Mirrors ``app.core.logging.get_logger``."""
    _configure()
    try:  # pragma: no cover - depends on the host's installed packages
        import structlog  # noqa: PLC0415

        return structlog.get_logger(name)
    except Exception:
        return _SDKLogger(name or "client")


__all__ = ["get_logger"]
