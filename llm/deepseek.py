from openai import AsyncOpenAI

from config import DEEPSEEK_KEY
from llm.base import LLMProvider
from llm.tool_loop import run_tool_loop
from llm.types import ChatOptions, ChatResult, ProviderCapabilities


class DeepSeekProvider(LLMProvider):
    def __init__(self, model: str):
        self.model = model
        self.client = AsyncOpenAI(
            api_key=DEEPSEEK_KEY,
            base_url="https://api.deepseek.com",
        )

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_thinking=True,
            thinking_toggle=True,
            supports_tool_search=True,
            thinking_search_exclusive=False,
        )

    def _extra_body(self, options: ChatOptions) -> dict:
        if options.enable_thinking:
            return {"thinking": {"type": "enabled"}}
        return {"thinking": {"type": "disabled"}}

    async def chat(
        self, user_text: str, *, system_prompt: str, options: ChatOptions
    ) -> ChatResult:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]
        extra_body = self._extra_body(options)
        max_tokens = 4096 if options.enable_thinking else 1024

        try:
            if options.enable_search:
                return await run_tool_loop(
                    self.client,
                    self.model,
                    messages,
                    max_tokens=max_tokens,
                    extra_body=extra_body,
                )

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                extra_body=extra_body,
            )
            message = response.choices[0].message
            reasoning = getattr(message, "reasoning_content", None)
            content = message.content or ""
            return ChatResult(content=content, reasoning=reasoning)
        except Exception as e:
            print(f"❌ 大模型调用失败: {e}")
            return ChatResult(
                content="抱歉主人，大模型接口调用失败了，请检查网络或API余额。"
            )
