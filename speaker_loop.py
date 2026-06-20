import asyncio
import json
import os

from asr import transcribe
from audio import (
    calibrate_noise,
    ensure_wake_audio,
    play_audio,
    record_audio,
    wait_for_wake_word,
)
from config import (
    REPLY_AUDIO_PATH,
    TEXT_DEBUG,
    WAKE_AUDIO_PATH,
    WAKE_MODEL_PATH,
    app_state,
    load_system_prompt,
)
from llm import get_provider
from tts import synthesize
import text_debug
from websocket_manager import ConnectionManager


async def smart_speaker_loop(manager: ConnectionManager):
    await asyncio.sleep(2)

    await ensure_wake_audio(WAKE_AUDIO_PATH)
    if TEXT_DEBUG:
        print("[TEXT_DEBUG] 跳过麦克风校准")
    else:
        calibrate_noise()

    while True:
        if TEXT_DEBUG:
            status = "💤 休眠中 (终端 /wake 或点「对话」)"
        else:
            status = "💤 休眠中 (喊 alexa 或点「对话」)"
        await manager.broadcast_status(status, "sleeping")
        await asyncio.to_thread(wait_for_wake_word, WAKE_MODEL_PATH)

        await manager.broadcast_status("✨ 我在！请说话...", "listening")
        await play_audio(WAKE_AUDIO_PATH)

        if TEXT_DEBUG:
            user_text = await text_debug.user_text_queue.get()
        else:
            wav_file = await asyncio.to_thread(record_audio)
            if wav_file is None:
                continue

            await manager.broadcast_status("语音识别中...", "transcribing")

            try:
                user_text = await asyncio.to_thread(transcribe, wav_file)
            finally:
                if wav_file and os.path.exists(wav_file):
                    os.remove(wav_file)

        if not user_text:
            continue

        if TEXT_DEBUG:
            await manager.broadcast_status("语音识别中...", "transcribing")

        await manager.broadcast(json.dumps({"type": "user_msg", "text": user_text}))

        try:
            provider = get_provider(app_state["model"])
            ai_reply = await provider.chat(user_text, system_prompt=load_system_prompt())
        except ValueError:
            ai_reply = "我不知道我在用什么模型..."

        await manager.broadcast_status("正在说话...", "speaking")
        await manager.broadcast(json.dumps({"type": "ai_msg", "text": ai_reply}))

        try:
            await synthesize(ai_reply, str(REPLY_AUDIO_PATH))
            await play_audio(REPLY_AUDIO_PATH)
        except Exception as e:
            print(f"❌ 语音合成或播放失败: {e}")
            await asyncio.sleep(2)
