from config import MAX_REPLY_CHARS

_SENTENCE_ENDS = "。！？；\n"


def truncate_reply(text: str, max_chars: int = MAX_REPLY_CHARS) -> str:
    if len(text) <= max_chars:
        return text

    chunk = text[:max_chars]
    last_break = max(chunk.rfind(c) for c in _SENTENCE_ENDS)
    if last_break > 0:
        return chunk[: last_break + 1]

    return chunk.rstrip() + "……"
