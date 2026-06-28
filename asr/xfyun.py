import base64
import hashlib
import hmac
import json
import ssl
import time
import wave
from email.utils import formatdate
from urllib.parse import urlencode

import websocket

from config import XFYUN_APP_ID, XFYUN_API_KEY, XFYUN_API_SECRET
from log_config import get_logger

logger = get_logger("asr.xfyun")

_HOST = "iat.xf-yun.com"
_PATH = "/v1"
_FRAME_SIZE = 1280
_FRAME_INTERVAL = 0.04
_MAX_PCM_BYTES = 16000 * 2 * 60
_EMPTY_PLACEHOLDERS = frozenset({"none", "null", "nan"})


def _create_auth_url() -> str:
    date = formatdate(timeval=None, localtime=False, usegmt=True)
    signature_origin = f"host: {_HOST}\ndate: {date}\nGET {_PATH} HTTP/1.1"
    signature_sha = hmac.new(
        XFYUN_API_SECRET.encode("utf-8"),
        signature_origin.encode("utf-8"),
        digestmod=hashlib.sha256,
    ).digest()
    signature = base64.b64encode(signature_sha).decode("utf-8")
    authorization_origin = (
        f'api_key="{XFYUN_API_KEY}", algorithm="hmac-sha256", '
        f'headers="host date request-line", signature="{signature}"'
    )
    authorization = base64.b64encode(authorization_origin.encode("utf-8")).decode(
        "utf-8"
    )
    query = urlencode({"authorization": authorization, "date": date, "host": _HOST})
    return f"wss://{_HOST}{_PATH}?{query}"


def _read_pcm_from_wav(path: str) -> bytes:
    with wave.open(path, "rb") as wf:
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
            raise ValueError(
                f"WAV 格式不符: channels={wf.getnchannels()} "
                f"width={wf.getsampwidth()} rate={wf.getframerate()}"
            )
        pcm = wf.readframes(wf.getnframes())
    if len(pcm) > _MAX_PCM_BYTES:
        logger.warning("音频超过 60s，截断至 60s")
        pcm = pcm[:_MAX_PCM_BYTES]
    return pcm


def _parse_result_text(b64_text: str) -> str:
    data = json.loads(base64.b64decode(b64_text))
    return "".join(
        cw.get("w", "")
        for ws in data.get("ws", [])
        for cw in ws.get("cw", [])
    )


def _first_frame(audio_b64: str, seq: int) -> dict:
    return {
        "header": {"app_id": XFYUN_APP_ID, "status": 0},
        "parameter": {
            "iat": {
                "domain": "slm",
                "language": "zh_cn",
                "accent": "mandarin",
                "result": {
                    "encoding": "utf8",
                    "compress": "raw",
                    "format": "json",
                },
            }
        },
        "payload": {
            "audio": {
                "encoding": "raw",
                "sample_rate": 16000,
                "channels": 1,
                "bit_depth": 16,
                "seq": seq,
                "status": 0,
                "audio": audio_b64,
            }
        },
    }


def _middle_frame(audio_b64: str, seq: int) -> dict:
    return {
        "header": {"app_id": XFYUN_APP_ID, "status": 1},
        "payload": {
            "audio": {
                "encoding": "raw",
                "sample_rate": 16000,
                "channels": 1,
                "bit_depth": 16,
                "seq": seq,
                "status": 1,
                "audio": audio_b64,
            }
        },
    }


def _last_frame(seq: int) -> dict:
    return {
        "header": {"app_id": XFYUN_APP_ID, "status": 2},
        "payload": {
            "audio": {
                "encoding": "raw",
                "sample_rate": 16000,
                "channels": 1,
                "bit_depth": 16,
                "seq": seq,
                "status": 2,
                "audio": "",
            }
        },
    }


def transcribe(wav_path: str) -> str:
    """调用讯飞中英识别大模型，返回识别文本；失败返回空字符串。"""
    if not XFYUN_APP_ID or not XFYUN_API_KEY or not XFYUN_API_SECRET:
        logger.error("讯飞 ASR 凭据未配置")
        return ""

    try:
        pcm = _read_pcm_from_wav(wav_path)
    except Exception as e:
        logger.error("读取 WAV 失败: %s", e)
        return ""

    if not pcm:
        return ""

    url = _create_auth_url()
    ws = None
    result_text = ""
    seq = 0

    try:
        ws = websocket.create_connection(
            url,
            sslopt={"cert_reqs": ssl.CERT_REQUIRED},
        )

        offset = 0
        while offset < len(pcm):
            chunk = pcm[offset : offset + _FRAME_SIZE]
            offset += _FRAME_SIZE
            seq += 1
            is_last = offset >= len(pcm)

            if seq == 1:
                frame = _first_frame(base64.b64encode(chunk).decode("ascii"), seq)
            else:
                frame = _middle_frame(base64.b64encode(chunk).decode("ascii"), seq)

            ws.send(json.dumps(frame))
            if not is_last:
                time.sleep(_FRAME_INTERVAL)

        seq += 1
        ws.send(json.dumps(_last_frame(seq)))

        while True:
            raw = ws.recv()
            if not raw:
                break

            msg = json.loads(raw)
            header = msg.get("header", {})
            code = header.get("code", -1)
            if code != 0:
                logger.error(
                    "讯飞 ASR 异常 code=%s message=%s",
                    code,
                    header.get("message", ""),
                )
                return ""

            payload = msg.get("payload") or {}
            result = payload.get("result") or {}
            b64_text = result.get("text")
            if b64_text:
                parsed = _parse_result_text(b64_text)
                if parsed:
                    result_text = parsed

            if header.get("status") == 2:
                break

        text = result_text.strip()
        if text.lower() in _EMPTY_PLACEHOLDERS:
            return ""
        return text
    except Exception as e:
        logger.error("讯飞 ASR 请求失败: %s", e)
        return ""
    finally:
        if ws is not None:
            ws.close()
