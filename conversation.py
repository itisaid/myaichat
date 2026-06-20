from config import app_state

MAX_HISTORY_TURNS = 8


def get_history() -> list[dict]:
    """Return a copy of recent user/assistant messages for LLM context."""
    return list(app_state["conversation_history"])


def append_turn(user_text: str, assistant_text: str) -> None:
    """Append one Q&A round and keep at most MAX_HISTORY_TURNS rounds."""
    history = app_state["conversation_history"]
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": assistant_text})
    max_messages = MAX_HISTORY_TURNS * 2
    if len(history) > max_messages:
        del history[: len(history) - max_messages]
