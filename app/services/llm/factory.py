"""
LLM provider factory + DB-backed config management — T37.

Priority chain for provider resolution:
  1. Explicit (provider, model) arguments
  2. DB default provider record (hot-swappable at runtime)
  3. Settings env vars (fallback)
"""
from __future__ import annotations

from functools import lru_cache

from app.core.config import get_settings
from app.core.exceptions import LLMError
from app.core.logging import get_logger
from app.services.llm.base import LLMProvider

logger = get_logger(__name__)


def _build_provider(provider: str, model: str | None = None, config: dict | None = None) -> LLMProvider:
    """
    Instantiate a provider adapter by name.
    `config` carries DB-level overrides (api_key, base_url, etc.).
    """
    cfg = config or {}

    if provider == "openai":
        from app.services.llm.openai_adapter import OpenAIAdapter
        return OpenAIAdapter(model=model, api_key=cfg.get("api_key"))

    if provider == "anthropic":
        from app.services.llm.anthropic_adapter import AnthropicAdapter
        return AnthropicAdapter(model=model, api_key=cfg.get("api_key"))

    if provider == "gemini":
        from app.services.llm.gemini_adapter import GeminiAdapter
        return GeminiAdapter(model=model, api_key=cfg.get("api_key"))

    if provider == "openrouter":
        from app.services.llm.openrouter_adapter import OpenRouterAdapter
        return OpenRouterAdapter(model=model, api_key=cfg.get("api_key"))

    if provider == "ollama":
        from app.services.llm.ollama_adapter import OllamaAdapter
        return OllamaAdapter(model=model, base_url=cfg.get("base_url"))

    if provider == "vllm":
        from app.services.llm.vllm_adapter import VLLMAdapter
        return VLLMAdapter(model=model, base_url=cfg.get("base_url"))

    raise LLMError(f"Unknown LLM provider: {provider!r}")


def get_llm_provider(
    provider: str | None = None,
    model: str | None = None,
) -> LLMProvider:
    """
    Synchronous factory — resolves the LLM provider from explicit args or env config.
    Does NOT hit the DB (use `get_llm_provider_from_db` for DB-backed resolution).
    """
    settings = get_settings()

    if provider:
        return _build_provider(provider, model=model)

    # Infer from env
    if settings.openai_api_key:
        return _build_provider("openai", model=model or settings.openai_default_model)
    if settings.anthropic_api_key:
        return _build_provider("anthropic", model=model or settings.anthropic_default_model)
    if settings.gemini_api_key:
        return _build_provider("gemini", model=model or settings.gemini_default_model)

    # Fall back to local Ollama
    return _build_provider("ollama", model=model or settings.ollama_default_model)


async def get_llm_provider_from_db(
    session,  # AsyncSession
    provider: str | None = None,
    model: str | None = None,
) -> LLMProvider:
    """
    DB-backed resolution — reads the active default Provider record.
    Falls back to env-based resolution if no DB record found.
    Supports hot-swapping the provider at runtime without restart.
    """
    from app.db.repositories import ProviderRepository

    repo = ProviderRepository(session)

    if provider:
        # Explicit provider name — still check DB for config overrides
        records = await repo.list_active(provider_type="llm")
        match = next((r for r in records if r.provider == provider), None)
        if match:
            return _build_provider(
                match.provider,
                model=model or match.model,
                config=match.configuration,
            )
        return _build_provider(provider, model=model)

    # No explicit provider — use DB default
    default_record = await repo.get_default(provider_type="llm")
    if default_record:
        logger.debug(
            "Using DB default provider",
            provider=default_record.provider,
            model=default_record.model,
        )
        return _build_provider(
            default_record.provider,
            model=model or default_record.model,
            config=default_record.configuration,
        )

    # Fall through to env-based
    logger.debug("No DB provider record found, falling back to env config")
    return get_llm_provider(model=model)
