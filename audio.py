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

from display import wake_display
from config import (
    MAX_HOLD_RECORD_SECONDS,
    MIN_RECORD_RMS,
    PHRASE_TIME_LIMIT,
    RECORD_START_TIMEOUT,
    TEXT_DEBUG,
    WAKE_KEYWORD_HITS,
    WAKE_MIC_WARMUP_SEC,
    WAKE_WORD_THRESHOLD,
    cancel_event,
    record_hold_event,
    wake_event,
)
from log_config import get_logger, suppress_native_stderr

logger = get_logger("audio")
recognizer = sr.Recognizer()

_wake_model: Model | None = None
_pa: pyaudio.PyAudio | None = None
_mic_stream = None
_downsample_factor = 1
_wake_audio_ready = False
_mic_needs_warmup = False


def _chunk_energy(buffer: bytes) -> float:
    if not buffer:
        return 0.0
    samples = np.frombuffer(buffer, dtype=np.int16)
    return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))


def _close_wake_mic():
    """关闭唤醒麦克风流；对话处理期间保持关闭，避免缓冲区堆积误唤醒。"""
    global _mic_stream, _pa
    if _mic_stream is not None:
        try:
            _mic_stream.stop_stream()
            _mic_stream.close()
        except Exception as e:
            logger.warning("关闭唤醒麦克风流失败: %s", e)
        _mic_stream = None
    if _pa is not None:
        try:
            _pa.terminate()
        except Exception as e:
            logger.warning("释放 PyAudio 失败: %s", e)
        _pa = None


def record_audio():
    record_hold_event.clear()
    logger.info("开始录音")

    try:
        with suppress_native_stderr():
            with sr.Microphone() as source:
                if source.stream is None:
                    logger.error("麦克风打开失败，stream 为 None")
                    return None

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
                    logger.warning("录音超时或音量过低，跳过本轮")
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
                    logger.warning("录音为空，跳过本轮")
                    return None
                rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
                if rms < MIN_RECORD_RMS:
                    logger.warning("录音音量过低 (rms=%.0f)，跳过本轮", rms)
                    return None

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                    tf.write(audio.get_wav_data(convert_rate=16000))
                    return tf.name
    except Exception as e:
        logger.error("录音失败: %s", e)
        return None


def calibrate_noise():
    logger.info("正在校准环境噪音，请保持安静 1 秒...")
    try:
        with suppress_native_stderr():
            with sr.Microphone() as global_source:
                recognizer.adjust_for_ambient_noise(global_source, duration=1)
        logger.info("噪声校准完成")
    except Exception as e:
        logger.error("麦克风初始化失败: %s", e)


def _open_mic_stream(pa: pyaudio.PyAudio):
    global _downsample_factor
    with suppress_native_stderr():
        try:
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=1280,
            )
            _downsample_factor = 1
            return stream
        except Exception:
            logger.warning("16000Hz 不可用，使用 48000Hz 兼容模式")
            try:
                stream = pa.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=48000,
                    input=True,
                    frames_per_buffer=1280 * 3,
                )
                _downsample_factor = 3
                return stream
            except Exception:
                logger.error("麦克风被占用或无法打开，请检查硬件")
                return None


def _ensure_wake_audio(model_path: str | os.PathLike) -> bool:
    global _wake_model, _pa, _mic_stream, _wake_audio_ready, _mic_needs_warmup

    if _wake_audio_ready and _mic_stream is not None:
        return True

    if _wake_model is None:
        model_path = os.fspath(model_path)
        logger.info("正在加载唤醒模型...")
        with suppress_native_stderr():
            try:
                _wake_model = Model(
                    wakeword_models=[model_path], inference_framework="onnx"
                )
            except TypeError:
                _wake_model = Model(wakeword_model_paths=[model_path])
        _wake_audio_ready = True
        logger.info("唤醒模型已加载")

    if _pa is None:
        with suppress_native_stderr():
            _pa = pyaudio.PyAudio()

    if _mic_stream is None:
        _mic_stream = _open_mic_stream(_pa)
        if _mic_stream is None:
            return False
        _mic_needs_warmup = True

    return True



def _read_wake_frame() -> np.ndarray:
    if _downsample_factor == 1:
        pcm = _mic_stream.read(1280, exception_on_overflow=False)
        return np.frombuffer(pcm, dtype=np.int16)
    pcm = _mic_stream.read(1280 * 3, exception_on_overflow=False)
    return np.frombuffer(pcm, dtype=np.int16)[::3]


def _warmup_wake_mic() -> None:
    global _mic_needs_warmup
    if not _mic_needs_warmup or _mic_stream is None or WAKE_MIC_WARMUP_SEC <= 0:
        _mic_needs_warmup = False
        return
    logger.info("唤醒麦克风预热 %.1fs", WAKE_MIC_WARMUP_SEC)
    deadline = time.time() + WAKE_MIC_WARMUP_SEC
    while time.time() < deadline:
        _read_wake_frame()
    _mic_needs_warmup = False

def wait_for_wake_word(model_path: str | os.PathLike):
    if TEXT_DEBUG:
        logger.debug("等待唤醒 (/wake、Enter 或界面「对话」)...")
        wake_event.clear()
        while True:
            if wake_event.is_set():
                logger.info("唤醒 source=text_debug")
                wake_display()
                return True
            time.sleep(0.1)

    if not _ensure_wake_audio(model_path):
        return False

    wake_event.clear()
    _warmup_wake_mic()
    hit_streak = 0

    while True:
        if wake_event.is_set():
            logger.info("唤醒 source=button")
            _close_wake_mic()
            wake_display()
            return True

        prediction = _wake_model.predict(_read_wake_frame())

        triggered = False
        for _mdl_name, score in prediction.items():
            if score > WAKE_WORD_THRESHOLD:
                hit_streak += 1
                if hit_streak >= WAKE_KEYWORD_HITS:
                    logger.info(
                        "唤醒 source=keyword score=%.2f hits=%d",
                        score,
                        hit_streak,
                    )
                    _close_wake_mic()
                    wake_display()
                    return True
                triggered = True
                break
        if not triggered:
            hit_streak = 0


async def play_audio(path: str | os.PathLike):
    if TEXT_DEBUG:
        logger.debug("播放 %s", path)
        return

    pygame.mixer.music.load(os.fspath(path))
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        if cancel_event.is_set():
            pygame.mixer.music.stop()
            return
        await asyncio.sleep(0.1)


def stop_playback():
    if TEXT_DEBUG:
        return
    if pygame.mixer.get_init():
        pygame.mixer.music.stop()


async def ensure_wake_audio(path: str | os.PathLike):
    path = os.fspath(path)
    if not os.path.exists(path):
        logger.info("正在预生成唤醒提示音...")
        tts = edge_tts.Communicate("我在", "zh-CN-XiaoxiaoNeural")
        await tts.save(path)
