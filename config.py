import os
from pathlib import Path

import dashscope
import pygame
from dotenv import load_dotenv

load_dotenv()

ALI_KEY = os.getenv("ALI_KEY")
DEEPSEEK_KEY = os.getenv("DEEPSEEK_KEY")
DEFAULT_MODEL = "deepseek-chat"

BASE_DIR = Path(__file__).resolve().parent
SYSTEM_PROMPT_PATH = BASE_DIR / "prompts" / "system.txt"
WAKE_MODEL_PATH = BASE_DIR / "alexa.onnx"
WAKE_AUDIO_PATH = BASE_DIR / "wozai.mp3"
REPLY_AUDIO_PATH = BASE_DIR / "reply.mp3"

app_state = {"model": DEFAULT_MODEL}

dashscope.api_key = ALI_KEY
pygame.mixer.init()


def load_system_prompt() -> str:
    """每次调用 LLM 前读取文件，修改 prompt 后下一轮对话即生效。"""
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
