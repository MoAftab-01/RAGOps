"""Provider resolution — the single place a backend is named.

RAGOps ships exactly one provider, local Ollama, because the project must run
with no paid API and no accounts. This module is the seam a second backend
would slot into, and it is deliberately the *only* place that would need to
change when one arrives: register the class in :data:`_PROVIDERS`, implement
:class:`app.providers.base.LLMProvider`, done. Every caller resolves through
:func:`get_provider` and never constructs an adapter directly, so adding a
cloud provider cannot require touching a service.

Two properties the rest of the app relies on:

* **No import-time I/O.** Importing this module opens no socket, reads no
  environment and touches no database. A provider is constructed on first
  :func:`get_provider` call, not at import.
* **One instance per process.** Adapters hold a pooled ``httpx`` client, so
  they are cached. :func:`reset_provider_cache` exists for tests and for the
  app lifespan after a settings change.
"""

from __future__ import annotations

from app.providers.base import LLMProvider
from app.providers.ollama import OllamaProvider

# Keyed by the name a caller passes to ``get_provider``. Ollama is first and is
# the default because it is the only provider the project requires.
_PROVIDERS: dict[str, type[LLMProvider]] = {
    OllamaProvider.name: OllamaProvider,
}

DEFAULT_PROVIDER: str = OllamaProvider.name

_provider_cache: dict[str, LLMProvider] = {}


def get_provider(name: str | None = None) -> LLMProvider:
    """Return the named provider, or the local Ollama provider by default.

    ``name`` is matched case-insensitively because it arrives from request
    bodies and config files, where ``"Ollama"`` and ``"ollama"`` are the same
    intent. An unknown name raises :class:`ValueError` listing what *is*
    registered — a silent fallback to Ollama would make a misconfigured cloud
    adapter look like it had worked, which is precisely the kind of thing this
    project must not do.
    """
    key = (name or DEFAULT_PROVIDER).strip().lower()
    if key not in _PROVIDERS:
        available = ", ".join(sorted(_PROVIDERS))
        raise ValueError(
            f"unknown LLM provider {name!r}; registered providers: {available}"
        )

    provider = _provider_cache.get(key)
    if provider is None:
        # Constructed here, not at import: building a provider must not open a
        # socket, and the app has to be importable with Ollama not running.
        provider = _PROVIDERS[key]()
        _provider_cache[key] = provider
    return provider


def available_providers() -> list[str]:
    """Names that can be passed to :func:`get_provider`, sorted."""
    return sorted(_PROVIDERS)


def reset_provider_cache() -> None:
    """Drop cached provider instances so the next call rebuilds them.

    For tests that change settings, and for the app lifespan after a
    configuration change. The previously cached adapters are not awaited shut
    down here — a caller that needs the transports closed should call
    :meth:`app.providers.base.LLMProvider.aclose` before resetting.
    """
    _provider_cache.clear()


def register_provider(
    name: str, provider_cls: type[LLMProvider]
) -> None:
    """Register an additional adapter class.

    Present so that adding a cloud backend is a one-line change in one file
    rather than a refactor. Nothing in RAGOps calls this today, and no cloud
    adapter is required to run the project.
    """
    key = name.strip().lower()
    if not key:
        raise ValueError("provider name must not be empty")
    _PROVIDERS[key] = provider_cls
    _provider_cache.pop(key, None)
