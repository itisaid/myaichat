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

from config import MIN_RECORD_RMS, wake_event

recognizer = sr.Recognizer()


def calibrate_noise():
    print("\n[系统提示] 🎤 正在校准环境基准噪音，请保持安静 1 秒钟...")
    try:
        with sr.Microphone() as global_source:
            recognizer.adjust_for_ambient_noise(global_source, duration=1)
        print("[系统提示] ✅ 噪声校准完成！设备已准备就绪。")
    except Exception as e:
        print(f"[错误] 麦克风初始化失败: {e}")


def wait_for_wake_word(model_path: str | os.PathLike):
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


def record_audio():
    with sr.Microphone() as source:
        try:
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=10)
        except sr.WaitTimeoutError:
            return None

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


async def play_audio(path: str | os.PathLike):
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
