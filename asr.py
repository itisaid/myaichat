import os
from typing import Any

from dashscope.audio.asr import Recognition

from log_config import get_logger

logger = get_logger("asr")

_EMPTY_PLACEHOLDERS = frozenset({"none", "null", "nan"})


def _extract_text(sentences: Any) -> str:
    if not sentences:
        return ""

    if isinstance(sentences, dict):
        return str(sentences.get("text", "") or "").strip()

    if isinstance(sentences, list):
        return "".join(
            str(s.get("text", "") or "")
            for s in sentences
            if isinstance(s, dict)
        ).strip()

    return ""


def transcribe(wav_path: str) -> str:
    """调用阿里 ASR，返回识别文本；失败返回空字符串。"""
    asr_instance = Recognition(
        model="paraformer-realtime-v2",
        format="wav",
        sample_rate=16000,
        callback=None,
    )

    try:
        response = asr_instance.call(os.path.abspath(wav_path))

        if response.status_code != 200:
            logger.error("阿里 ASR 异常 status=%s", response.status_code)
            return ""

        if hasattr(response, "get_sentence"):
            sentences = response.get_sentence()
        else:
            sentences = response.output.get("sentences", [])

        text = _extract_text(sentences)
        if text.lower() in _EMPTY_PLACEHOLDERS:
            return ""
        return text
    except Exception as e:
        logger.error("语音识别请求失败: %s", e)
        return ""
