from abc import ABC, abstractmethod


class LLMProvider(ABC):
    @abstractmethod
    async def chat(self, user_text: str, *, system_prompt: str) -> str: ...
