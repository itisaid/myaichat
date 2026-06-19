from openai import AsyncOpenAI

from config import DEEPSEEK_KEY
from llm.base import LLMProvider


class DeepSeekProvider(LLMProvider):
    def __init__(self, model: str):
        self.model = model
        self.client = AsyncOpenAI(
            api_key=DEEPSEEK_KEY,
            base_url="https://api.deepseek.com",
        )

    async def chat(self, user_text: str, *, system_prompt: str) -> str:
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text},
                ],
                max_tokens=100,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"❌ 大模型调用失败: {e}")
            return "抱歉主人，大模型接口调用失败了，请检查网络或API余额。"
