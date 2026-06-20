from llm.base import LLMProvider
from llm.types import ChatOptions, ChatResult, ProviderCapabilities


class QwenProvider(LLMProvider):
    """通义千问适配器（占位）。接入时映射 enable_thinking / enable_search extra_body。"""

    def __init__(self, model: str):
        self.model = model

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_thinking=True,
            thinking_toggle=True,
            supports_native_search=True,
            supports_tool_search=True,
            thinking_search_exclusive=False,
        )

    async def chat(
        self, user_text: str, *, system_prompt: str, options: ChatOptions
    ) -> ChatResult:
        # TODO: dashscope OpenAI 兼容模式
        # extra_body={"enable_thinking": options.enable_thinking,
        #             "enable_search": options.enable_search}
        return ChatResult(content="（来自通义千问）很高兴为您服务，今天阳光明媚。")
