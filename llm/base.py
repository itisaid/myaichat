from abc import ABC, abstractmethod

from llm.types import ChatOptions, ChatResult, ProviderCapabilities


class LLMProvider(ABC):
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities()

    @abstractmethod
    async def chat(
        self, user_text: str, *, system_prompt: str, options: ChatOptions
    ) -> ChatResult: ...
