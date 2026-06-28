import os
import threading
from pathlib import Path

import dashscope
import pygame
from dotenv import load_dotenv

load_dotenv()

ALI_KEY = os.getenv("ALI_KEY")
DEEPSEEK_KEY = os.getenv("DEEPSEEK_KEY")
XFYUN_APP_ID = os.getenv("XFYUN_APP_ID")
XFYUN_API_KEY = os.getenv("XFYUN_API_KEY")
XFYUN_API_SECRET = os.getenv("XFYUN_API_SECRET")
ASR_PROVIDER = os.getenv("ASR_PROVIDER", "xfyun")
DASHSCOPE_BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)
DEFAULT_MODEL = "qwen-plus"
TEXT_DEBUG = os.getenv("TEXT_DEBUG", "").lower() in ("1", "true", "yes")
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG" if TEXT_DEBUG else "INFO")
DISPLAY_WAKE_ENABLED = os.getenv("DISPLAY_WAKE_ENABLED", "1") == "1"
TOUCH_MONITOR_ENABLED = os.getenv("TOUCH_MONITOR_ENABLED", "1") == "1"
DISPLAY_WAKE_USE_XDOTOOL = os.getenv("DISPLAY_WAKE_USE_XDOTOOL", "0") == "1"

BASE_DIR = Path(__file__).resolve().parent
SYSTEM_PROMPT_PATH = BASE_DIR / "prompts" / "system.txt"
WAKE_MODEL_PATH = BASE_DIR / "alexa.onnx"
WAKE_AUDIO_PATH = BASE_DIR / "wozai.mp3"
WAKE_WORD_THRESHOLD = 0.5
WAKE_MIC_WARMUP_SEC = 1.5
WAKE_KEYWORD_HITS = 3
WAKE_POST_PLAYBACK_COOLDOWN = 1.5
REPLY_AUDIO_PATH = BASE_DIR / "reply.mp3"
MIN_RECORD_RMS = 400
PHRASE_TIME_LIMIT = 10
RECORD_START_TIMEOUT = 10
MAX_HOLD_RECORD_SECONDS = 60
MAX_REPLY_CHARS = 800
MAX_REPLY_TOKENS = 600
MAX_REPLY_TOKENS_THINKING = 2048

app_state = {
    "model": DEFAULT_MODEL,
    "enable_thinking": False,
    "enable_search": False,
    "status_text": "系统启动中...",
    "phase": "starting",
    "wake_enabled": False,
    "stop_enabled": False,
    "conversation_history": [],
}
wake_event = threading.Event()
record_hold_event = threading.Event()
cancel_event = threading.Event()

dashscope.api_key = ALI_KEY
if not TEXT_DEBUG:
    pygame.mixer.init()


def load_system_prompt() -> str:
    """每次调用 LLM 前读取文件，修改 prompt 后下一轮对话即生效。"""
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
