"""LLM access.

One provider is required — local Ollama, free and account-free — and the
abstraction exists so a second one could be added without touching a service.
Resolve providers through :func:`app.providers.registry.get_provider` rather
than constructing them directly, so the cache and the registry stay
authoritative.
"""

from __future__ import annotations

from app.providers.base import (
    ChatMessage,
    LLMProvider,
    LLMProviderError,
    LLMResponse,
)
from app.providers.ollama import OllamaProvider
from app.providers.registry import (
    DEFAULT_PROVIDER,
    available_providers,
    get_provider,
    register_provider,
    reset_provider_cache,
)

__all__ = [
    "DEFAULT_PROVIDER",
    "ChatMessage",
    "LLMProvider",
    "LLMProviderError",
    "LLMResponse",
    "OllamaProvider",
    "available_providers",
    "get_provider",
    "register_provider",
    "reset_provider_cache",
]
