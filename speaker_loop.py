import asyncio
import json
import os

from asr import transcribe
from audio import (
    calibrate_noise,
    ensure_wake_audio,
    play_audio,
    record_audio,
    stop_playback,
    wait_for_wake_word,
)
from config import (
    REPLY_AUDIO_PATH,
    TEXT_DEBUG,
    WAKE_AUDIO_PATH,
    WAKE_MODEL_PATH,
    app_state,
    cancel_event,
    load_system_prompt,
)
from llm import get_provider
from llm.truncate import truncate_reply
from llm.types import ChatOptions, ChatResult
from tts import synthesize
import text_debug
from websocket_manager import ConnectionManager


def _sleeping_status() -> str:
    if TEXT_DEBUG:
        return "💤 休眠中 (终端 /wake 或点「对话」)"
    return "💤 休眠中 (喊 alexa 或点「对话」)"


async def abort_to_sleeping(manager: ConnectionManager):
    cancel_event.clear()
    stop_playback()
    await manager.broadcast_status(_sleeping_status(), "sleeping", stop_enabled=False)


class _Cancelled:
    """Sentinel: await_cancellable was interrupted by cancel_event."""


CANCELLED = _Cancelled()


async def await_cancellable(awaitable):
    task = (
        awaitable
        if isinstance(awaitable, asyncio.Task)
        else asyncio.create_task(awaitable)
    )
    while not task.done():
        if cancel_event.is_set():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            return CANCELLED
        await asyncio.sleep(0.05)
    try:
        return await task
    except asyncio.CancelledError:
        return CANCELLED


async def smart_speaker_loop(manager: ConnectionManager):
    await asyncio.sleep(2)

    await ensure_wake_audio(WAKE_AUDIO_PATH)
    if TEXT_DEBUG:
        print("[TEXT_DEBUG] 跳过麦克风校准")
    else:
        calibrate_noise()

    while True:
        await manager.broadcast_status(_sleeping_status(), "sleeping", stop_enabled=False)
        await asyncio.to_thread(wait_for_wake_word, WAKE_MODEL_PATH)

        await manager.broadcast_status(
            "✨ 我在！请说话...", "listening", stop_enabled=False
        )
        await play_audio(WAKE_AUDIO_PATH)
        if cancel_event.is_set():
            await abort_to_sleeping(manager)
            continue

        if TEXT_DEBUG:
            user_text = await text_debug.user_text_queue.get()
        else:
            wav_file = await asyncio.to_thread(record_audio)
            if wav_file is None:
                continue

            await manager.broadcast_status(
                "语音识别中...", "transcribing", stop_enabled=True
            )

            try:
                user_text = await await_cancellable(
                    asyncio.to_thread(transcribe, wav_file)
                )
            finally:
                if wav_file and os.path.exists(wav_file):
                    os.remove(wav_file)

            if user_text is CANCELLED:
                await abort_to_sleeping(manager)
                continue

        if not user_text:
            continue

        if TEXT_DEBUG:
            await manager.broadcast_status(
                "语音识别中...", "transcribing", stop_enabled=True
            )

        await manager.broadcast(json.dumps({"type": "user_msg", "text": user_text}))

        options = ChatOptions(
            enable_thinking=app_state["enable_thinking"],
            enable_search=app_state["enable_search"],
        )

        if options.enable_search:
            await manager.broadcast(
                json.dumps({"type": "search_status", "status": "pending"})
            )

        await manager.broadcast_status("思考中...", "transcribing", stop_enabled=True)

        try:
            provider = get_provider(app_state["model"])
            result = await await_cancellable(
                provider.chat(
                    user_text, system_prompt=load_system_prompt(), options=options
                )
            )
        except ValueError:
            result = ChatResult(content="我不知道我在用什么模型...")

        if result is CANCELLED:
            await abort_to_sleeping(manager)
            continue

        if options.enable_search:
            await manager.broadcast(
                json.dumps(
                    {
                        "type": "search_status",
                        "status": result.search_status,
                    }
                )
            )

        if result.reasoning:
            await manager.broadcast(
                json.dumps({"type": "thinking_msg", "text": result.reasoning})
            )

        reply_text = truncate_reply(result.content)
        if len(reply_text) < len(result.content):
            print(
                f"⚠️ 回答已截断: {len(result.content)} 字 -> {len(reply_text)} 字"
            )

        await manager.broadcast_status("正在说话...", "speaking", stop_enabled=True)
        await manager.broadcast(json.dumps({"type": "ai_msg", "text": reply_text}))

        try:
            if (
                await await_cancellable(
                    synthesize(reply_text, str(REPLY_AUDIO_PATH))
                )
                is CANCELLED
            ):
                await abort_to_sleeping(manager)
                continue

            await play_audio(REPLY_AUDIO_PATH)
            if cancel_event.is_set():
                await abort_to_sleeping(manager)
                continue
        except Exception as e:
            print(f"❌ 语音合成或播放失败: {e}")
            await asyncio.sleep(2)
