import asyncio

from duckduckgo_search import DDGS

from log_config import get_logger

logger = get_logger("search")

SEARCH_TIMEOUT = 8.0
SEARCH_FALLBACK = "搜索未返回结果，请基于已有知识回答。"


async def web_search(query: str, timeout: float = SEARCH_TIMEOUT) -> tuple[str | None, str]:
    """Return (result_text, status) where status is success | failed | timeout."""
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(_search_sync, query),
            timeout=timeout,
        )
        if result:
            return result, "success"
        return None, "failed"
    except asyncio.TimeoutError:
        logger.warning("联网搜索超时: %s", query)
        return None, "timeout"
    except Exception as e:
        logger.error("联网搜索失败: %s", e)
        return None, "failed"


def _search_sync(query: str) -> str | None:
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=5))
    if not results:
        return None
    parts = []
    for item in results:
        title = item.get("title", "")
        body = item.get("body", "")
        parts.append(f"{title}: {body}")
    return "\n\n".join(parts)
