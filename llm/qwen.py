from openai import AsyncOpenAI

from config import (
    ALI_KEY,
    DASHSCOPE_BASE_URL,
    MAX_REPLY_TOKENS,
    MAX_REPLY_TOKENS_THINKING,
)
from llm.base import LLMProvider
from llm.types import ChatOptions, ChatResult, ProviderCapabilities


class QwenProvider(LLMProvider):
    """通义千问适配器，通过百炼 OpenAI 兼容接口调用。"""

    def __init__(self, model: str):
        self.model = model
        self.client = AsyncOpenAI(
            api_key=ALI_KEY,
            base_url=DASHSCOPE_BASE_URL,
        )

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_thinking=True,
            thinking_toggle=True,
            supports_native_search=True,
            supports_tool_search=True,
            thinking_search_exclusive=False,
        )

    def _extra_body(self, options: ChatOptions) -> dict:
        body = {"enable_thinking": options.enable_thinking}
        if options.enable_search:
            body["enable_search"] = True
        return body

    async def chat(
        self,
        user_text: str,
        *,
        system_prompt: str,
        options: ChatOptions,
        history: list[dict] | None = None,
    ) -> ChatResult:
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_text})
        extra_body = self._extra_body(options)
        max_tokens = (
            MAX_REPLY_TOKENS_THINKING if options.enable_thinking else MAX_REPLY_TOKENS
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                extra_body=extra_body,
            )
            message = response.choices[0].message
            reasoning = getattr(message, "reasoning_content", None)
            content = message.content or ""
            search_status = "success" if options.enable_search else "none"
            return ChatResult(
                content=content,
                reasoning=reasoning,
                search_status=search_status,
            )
        except Exception as e:
            print(f"❌ 大模型调用失败: {e}")
            return ChatResult(
                content="抱歉主人，大模型接口调用失败了，请检查网络或API余额。",
                search_status="failed" if options.enable_search else "none",
            )
