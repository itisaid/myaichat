from llm.base import LLMProvider


class QwenProvider(LLMProvider):
    def __init__(self, model: str):
        self.model = model

    async def chat(self, user_text: str, *, system_prompt: str) -> str:
        # TODO: 使用 dashscope 库或 openai 库 + 阿里百炼 API Key
        return "（来自通义千问）很高兴为您服务，今天阳光明媚。"
