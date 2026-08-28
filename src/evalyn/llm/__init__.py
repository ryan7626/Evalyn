"""LLM provider construction for Evalyn."""

from __future__ import annotations

from evalyn.llm.base import BaseLLM
from evalyn.llm.claude_provider import ClaudeProvider
from evalyn.llm.openai_compatible_provider import OpenAICompatibleProvider
from evalyn.settings import ProviderSettings


def create_provider(
    settings: ProviderSettings,
    *,
    timeout_seconds: float = 120,
) -> BaseLLM:
    if settings.kind == "anthropic":
        return ClaudeProvider(model=settings.model, timeout_seconds=timeout_seconds)
    if settings.kind == "openai-compatible":
        return OpenAICompatibleProvider(
            model=settings.model,
            base_url=settings.base_url,
            timeout_seconds=timeout_seconds,
        )
    raise ValueError(f"Unsupported provider: {settings.kind}")


__all__ = ["BaseLLM", "create_provider"]
