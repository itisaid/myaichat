from llm.base import LLMProvider
from llm.types import ChatOptions, ChatResult, ProviderCapabilities


class GLMProvider(LLMProvider):
    """智谱 GLM 适配器（占位）。接入时映射 enable_thinking / FC 搜索。"""

    def __init__(self, model: str):
        self.model = model

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_thinking=True,
            thinking_toggle=True,
            supports_tool_search=True,
            thinking_search_exclusive=True,
        )

    async def chat(
        self,
        user_text: str,
        *,
        system_prompt: str,
        options: ChatOptions,
        history: list[dict] | None = None,
    ) -> ChatResult:
        # TODO: zhipuai OpenAI 兼容模式
        # extra_body={"enable_thinking": options.enable_thinking}
        return ChatResult(content="（来自智谱GLM）今天天气不错，适合出门哦。")
