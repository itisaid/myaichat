import asyncio
import os
import tempfile
import time

import edge_tts
import numpy as np
import pyaudio
import pygame
import speech_recognition as sr
from openwakeword.model import Model

from config import (
    MAX_HOLD_RECORD_SECONDS,
    MIN_RECORD_RMS,
    PHRASE_TIME_LIMIT,
    RECORD_START_TIMEOUT,
    TEXT_DEBUG,
    record_hold_event,
    wake_event,
)

recognizer = sr.Recognizer()


def _chunk_energy(buffer: bytes) -> float:
    if not buffer:
        return 0.0
    samples = np.frombuffer(buffer, dtype=np.int16)
    return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))


def record_audio():
    record_hold_event.clear()

    with sr.Microphone() as source:
        chunk = source.CHUNK
        sample_width = source.SAMPLE_WIDTH
        sample_rate = source.SAMPLE_RATE

        frames: list[bytes] = []
        wait_deadline = time.time() + RECORD_START_TIMEOUT
        while time.time() < wait_deadline:
            buffer = source.stream.read(chunk)
            if _chunk_energy(buffer) >= recognizer.energy_threshold:
                frames.append(buffer)
                break
        else:
            return None

        last_speech_time = time.time()
        record_start = time.time()
        hold_was_active = False

        while True:
            hold_active = record_hold_event.is_set()
            if hold_was_active and not hold_active:
                break
            hold_was_active = hold_active

            now = time.time()
            elapsed = now - record_start

            if elapsed >= MAX_HOLD_RECORD_SECONDS:
                break

            if not hold_active:
                if elapsed >= PHRASE_TIME_LIMIT:
                    break
                if now - last_speech_time >= recognizer.pause_threshold:
                    break

            buffer = source.stream.read(chunk)
            frames.append(buffer)
            if _chunk_energy(buffer) >= recognizer.energy_threshold:
                last_speech_time = time.time()

        audio = sr.AudioData(b"".join(frames), sample_rate, sample_width)
        pcm = audio.get_raw_data(convert_rate=16000, convert_width=2)
        samples = np.frombuffer(pcm, dtype=np.int16)
        if samples.size == 0:
            return None
        rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
        if rms < MIN_RECORD_RMS:
            return None

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            tf.write(audio.get_wav_data(convert_rate=16000))
            return tf.name


def calibrate_noise():
    print("\n[系统提示] 🎤 正在校准环境基准噪音，请保持安静 1 秒钟...")
    try:
        with sr.Microphone() as global_source:
            recognizer.adjust_for_ambient_noise(global_source, duration=1)
        print("[系统提示] ✅ 噪声校准完成！设备已准备就绪。")
    except Exception as e:
        print(f"[错误] 麦克风初始化失败: {e}")


def wait_for_wake_word(model_path: str | os.PathLike):
    if TEXT_DEBUG:
        print("\n💤 [TEXT_DEBUG] 等待唤醒 (/wake、Enter 或界面「对话」)...")
        wake_event.clear()
        while True:
            if wake_event.is_set():
                print("\n🔔 [唤醒] 终端或界面触发对话")
                return True
            time.sleep(0.1)

    print("\n[系统状态] 正在加载唤醒模型，请稍候...")
    model_path = os.fspath(model_path)

    try:
        oww_model = Model(wakeword_models=[model_path], inference_framework="onnx")
    except TypeError:
        oww_model = Model(wakeword_model_paths=[model_path])

    print("[系统状态] 模型加载完成，准备接管麦克风...")
    time.sleep(1)

    pa = pyaudio.PyAudio()

    try:
        mic_stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=1280,
        )
        downsample_factor = 1
    except Exception:
        print("⚠️ 麦克风不支持 16000Hz，自动开启 48000Hz 兼容模式...")
        try:
            mic_stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=48000,
                input=True,
                frames_per_buffer=1280 * 3,
            )
            downsample_factor = 3
        except Exception:
            print("❌ [错误] 麦克风被占用或彻底无法打开，请检查硬件！")
            return False

    print("\n💤 [休眠中] 等待唤醒词...")
    wake_event.clear()

    try:
        while True:
            if wake_event.is_set():
                print("\n🔔 [唤醒] 界面按钮触发对话")
                return True

            if downsample_factor == 1:
                pcm = mic_stream.read(1280, exception_on_overflow=False)
                audio_data = np.frombuffer(pcm, dtype=np.int16)
            else:
                pcm = mic_stream.read(1280 * 3, exception_on_overflow=False)
                audio_data = np.frombuffer(pcm, dtype=np.int16)[::3]

            prediction = oww_model.predict(audio_data)

            for _mdl_name, score in prediction.items():
                if score > 0.2:
                    print(f"\n🔔 [唤醒] 检测到唤醒词！(置信度: {score:.2f})")
                    return True
    finally:
        mic_stream.stop_stream()
        mic_stream.close()
        pa.terminate()


async def play_audio(path: str | os.PathLike):
    if TEXT_DEBUG:
        print(f"[播放] {path}")
        return

    pygame.mixer.music.load(os.fspath(path))
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        await asyncio.sleep(0.1)


async def ensure_wake_audio(path: str | os.PathLike):
    path = os.fspath(path)
    if not os.path.exists(path):
        print("\n[系统状态] 正在预生成唤醒提示音...")
        tts = edge_tts.Communicate("我在", "zh-CN-XiaoxiaoNeural")
        await tts.save(path)
