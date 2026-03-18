"""Provider registry and factory functions."""

from typing import Optional

from providers.base import BaseLLMProvider, BaseEmbedder


_LLM_REGISTRY = {
    "openai": ("providers.llm_openai", "OpenAILLMProvider"),
    "anthropic": ("providers.llm_anthropic", "AnthropicLLMProvider"),
}

_EMBEDDER_REGISTRY = {
    "openai": ("providers.embedder_openai", "OpenAIEmbedder"),
}


def _import_class(module_path: str, class_name: str):
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def get_llm_provider(provider_name: Optional[str] = None) -> BaseLLMProvider:
    from config import Config

    if provider_name is None:
        provider_name = Config.LLM_PROVIDER
    provider_name = provider_name.lower()

    if provider_name not in _LLM_REGISTRY:
        available = ", ".join(sorted(_LLM_REGISTRY.keys()))
        raise ValueError(
            f"Unknown LLM provider: '{provider_name}'. "
            f"Supported providers: {available}"
        )

    module_path, class_name = _LLM_REGISTRY[provider_name]
    provider_class = _import_class(module_path, class_name)

    if provider_name == "openai":
        return provider_class(api_key=Config.OPENAI_API_KEY)
    elif provider_name == "anthropic":
        return provider_class(api_key=Config.ANTHROPIC_API_KEY)
    else:
        return provider_class()


def get_embedder(provider_name: Optional[str] = None) -> BaseEmbedder:
    from config import Config

    if provider_name is None:
        provider_name = Config.EMBEDDING_PROVIDER
    provider_name = provider_name.lower()

    if provider_name not in _EMBEDDER_REGISTRY:
        available = ", ".join(sorted(_EMBEDDER_REGISTRY.keys()))
        raise ValueError(
            f"Unknown embedding provider: '{provider_name}'. "
            f"Supported providers: {available}"
        )

    module_path, class_name = _EMBEDDER_REGISTRY[provider_name]
    embedder_class = _import_class(module_path, class_name)

    if provider_name == "openai":
        return embedder_class(
            api_key=Config.OPENAI_API_KEY,
            model=Config.EMBEDDING_MODEL,
            dimension=Config.EMBEDDING_DIM,
        )
    else:
        return embedder_class()


def list_llm_providers() -> dict:
    from config import Config

    available_set = set(Config.get_available_providers())
    result = {}
    for name in _LLM_REGISTRY:
        result[name] = {
            "available": name in available_set,
            "module": _LLM_REGISTRY[name][0],
        }
    return result


def list_embedder_providers() -> dict:
    from config import Config

    result = {}
    for name in _EMBEDDER_REGISTRY:
        available = bool(Config.OPENAI_API_KEY) if name == "openai" else True
        result[name] = {
            "available": available,
            "module": _EMBEDDER_REGISTRY[name][0],
        }
    return result
