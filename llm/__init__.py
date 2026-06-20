from llm.base import LLMProvider
from llm.deepseek import DeepSeekProvider
from llm.glm import GLMProvider
from llm.qwen import QwenProvider
from llm.types import ProviderCapabilities

PROVIDERS = {
    "deepseek": DeepSeekProvider,
    "qwen": QwenProvider,
    "glm": GLMProvider,
}


def get_provider(model_name: str) -> LLMProvider:
    for key, cls in PROVIDERS.items():
        if key in model_name:
            return cls(model=model_name)
    raise ValueError(f"Unknown model: {model_name}")


def get_capabilities(model_name: str) -> dict:
    try:
        return get_provider(model_name).capabilities().to_dict()
    except ValueError:
        return ProviderCapabilities().to_dict()
