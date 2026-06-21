import logging
import os
import sys
import warnings
from contextlib import contextmanager
from logging import Logger

_CONFIGURED = False


def _resolve_log_level() -> int:
    text_debug = os.getenv("TEXT_DEBUG", "").lower() in ("1", "true", "yes")
    level_name = os.getenv("LOG_LEVEL", "DEBUG" if text_debug else "INFO").upper()
    return getattr(logging, level_name, logging.INFO)


class _BracketFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        record.bracket_name = f"[{record.name}]"
        return super().format(record)


def setup_logging() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    level = _resolve_log_level()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        _BracketFormatter(
            fmt="%(asctime)s %(levelname)-5s %(bracket_name)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    for name, logger_level in (
        ("uvicorn.access", logging.WARNING),
        ("uvicorn.error", logging.INFO),
        ("onnxruntime", logging.ERROR),
    ):
        logging.getLogger(name).setLevel(logger_level)

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings(
        "ignore",
        message="Specified provider 'CUDAExecutionProvider'.*",
        category=UserWarning,
    )

    _CONFIGURED = True


def get_logger(name: str) -> Logger:
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(name)


@contextmanager
def suppress_native_stderr():
    """Redirect fd=2 during native audio init to silence ALSA/JACK noise."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        os.dup2(old_stderr, 2)
        os.close(devnull)
        os.close(old_stderr)


def truncate_text(text: str, max_len: int = 80) -> str:
    text = text.replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"
