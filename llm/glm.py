from llm.base import LLMProvider


class GLMProvider(LLMProvider):
    def __init__(self, model: str):
        self.model = model

    async def chat(self, user_text: str, *, system_prompt: str) -> str:
        # TODO: 使用 zhipuai 库 + 智谱 API Key
        return "（来自智谱GLM）今天天气不错，适合出门哦。"
