from config import ASR_PROVIDER
from log_config import get_logger

logger = get_logger("asr")

_ASR_LABELS = {
    "xfyun": "讯飞中英识别大模型",
    "ali": "阿里 paraformer-realtime-v2",
}


def transcribe(wav_path: str) -> str:
    """语音识别入口，按 ASR_PROVIDER 路由到对应服务。"""
    provider = ASR_PROVIDER if ASR_PROVIDER in _ASR_LABELS else "xfyun"
    if provider == "ali":
        from asr.ali import transcribe as _transcribe
    else:
        from asr.xfyun import transcribe as _transcribe

    text = _transcribe(wav_path)
    if text:
        logger.info("ASR 服务=%s", _ASR_LABELS[provider])
    return text
