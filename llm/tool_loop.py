import asyncio
import json
from collections.abc import Awaitable, Callable

from openai import AsyncOpenAI

from config import cancel_event
from llm.types import ChatResult
from search.duckduckgo import SEARCH_FALLBACK, web_search

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "搜索互联网获取最新信息，适用于新闻、天气、实时数据等问题。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词",
                }
            },
            "required": ["query"],
        },
    },
}

ToolHandler = Callable[[str, dict], Awaitable[tuple[str, str]]]


async def _default_web_search_handler(_name: str, args: dict) -> tuple[str, str]:
    query = args.get("query", "")
    if not query:
        return SEARCH_FALLBACK, "failed"
    text, status = await web_search(query)
    if text:
        return text, status
    return SEARCH_FALLBACK, status


async def run_tool_loop(
    client: AsyncOpenAI,
    model: str,
    messages: list[dict],
    *,
    max_tokens: int = 4096,
    extra_body: dict | None = None,
    max_rounds: int = 5,
    tool_handler: ToolHandler | None = None,
) -> ChatResult:
    handler = tool_handler or _default_web_search_handler
    tools = [WEB_SEARCH_TOOL]
    search_queries: list[str] = []
    search_status = "none"
    reasoning_parts: list[str] = []

    for _ in range(max_rounds):
        if cancel_event.is_set():
            raise asyncio.CancelledError()

        kwargs: dict = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "max_tokens": max_tokens,
        }
        if extra_body:
            kwargs["extra_body"] = extra_body

        response = await client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        reasoning = getattr(message, "reasoning_content", None)
        if reasoning:
            reasoning_parts.append(reasoning)

        if finish_reason != "tool_calls" or not message.tool_calls:
            content = message.content or ""
            return ChatResult(
                content=content,
                reasoning="\n\n".join(reasoning_parts) or None,
                search_queries=search_queries,
                search_status=search_status,
            )

        # Assistant message for tool round: omit reasoning_content (DeepSeek API constraint).
        messages.append(
            {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ],
            }
        )

        for tool_call in message.tool_calls:
            fn_name = tool_call.function.name
            try:
                fn_args = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                fn_args = {}

            tool_content, tool_status = await handler(fn_name, fn_args)

            if fn_name == "web_search":
                query = fn_args.get("query", "")
                if query:
                    search_queries.append(query)
                if tool_status == "success":
                    search_status = "success"
                elif search_status != "success":
                    search_status = tool_status

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_content,
                }
            )

    return ChatResult(
        content="抱歉，处理您的请求时步骤过多，请换个方式提问。",
        reasoning="\n\n".join(reasoning_parts) or None,
        search_queries=search_queries,
        search_status=search_status,
    )
