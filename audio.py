import asyncio
import os
import tempfile
import threading
import time
import wave

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
    WAKE_PRE_LISTEN_DRAIN_SEC,
    WAKE_WORD_THRESHOLD,
    RECORD_ENERGY_THRESHOLD_MAX,
    cancel_event,
    record_hold_event,
    wake_event,
)
from log_config import get_logger, suppress_native_stderr

logger = get_logger("audio")
recognizer = sr.Recognizer()

MIC_SAMPLE_RATE = 16000
MIC_SAMPLE_WIDTH = 2
FRAME_SAMPLES = 1280

_wake_model: Model | None = None
_pa: pyaudio.PyAudio | None = None
_mic_stream = None
_downsample_factor = 1
_wake_audio_ready = False
_wake_listen_enabled = True
_mic_lock = threading.Lock()
_drain_thread: threading.Thread | None = None
_drain_stop = threading.Event()


def _chunk_energy(buffer: bytes) -> float:
    if not buffer:
        return 0.0
    samples = np.frombuffer(buffer, dtype=np.int16)
    return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))


def reset_wake_model():
    """清空唤醒模型内部音频/预测缓冲，避免上次唤醒词残留导致立即误唤醒。"""
    if _wake_model is not None and hasattr(_wake_model, "reset"):
        try:
            _wake_model.reset()
        except Exception as e:
            logger.warning("唤醒模型 reset 失败: %s", e)


def pause_wake_listen():
    """对话期间停止唤醒检测，后台 drain 防止缓冲区堆积。"""
    global _wake_listen_enabled
    _wake_listen_enabled = False
    _start_drain_thread()


def resume_wake_listen():
    """回到休眠监听（不 drain，供 abort 等路径快速恢复）。"""
    _stop_drain_thread()
    reset_wake_model()
    global _wake_listen_enabled
    _wake_listen_enabled = True


def prepare_wake_listen(drain_sec: float | None = None):
    """丢弃残留音频后再开启唤醒检测。"""
    global _wake_listen_enabled
    _wake_listen_enabled = False
    sec = WAKE_PRE_LISTEN_DRAIN_SEC if drain_sec is None else drain_sec
    drain_mic(sec)
    reset_wake_model()
    _wake_listen_enabled = True


def drain_mic(seconds: float = 0):
    """同步丢弃麦克风音频；会先停止后台 drain 线程。"""
    _stop_drain_thread()
    if seconds <= 0 or _mic_stream is None:
        return
    logger.info("麦克风 drain %.1fs", seconds)
    deadline = time.time() + seconds
    while time.time() < deadline:
        _read_mic_frame()


def _downsample_frame(pcm: bytes) -> np.ndarray:
    samples = np.frombuffer(pcm, dtype=np.int16)
    if _downsample_factor == 1:
        return samples
    return samples.reshape(-1, _downsample_factor).mean(axis=1).astype(np.int16)


def _read_mic_frame() -> np.ndarray:
    with _mic_lock:
        if _mic_stream is None:
            return np.array([], dtype=np.int16)
        if _downsample_factor == 1:
            pcm = _mic_stream.read(FRAME_SAMPLES, exception_on_overflow=False)
        else:
            pcm = _mic_stream.read(
                FRAME_SAMPLES * _downsample_factor, exception_on_overflow=False
            )
        return _downsample_frame(pcm)


def _read_mic_bytes() -> bytes:
    return _read_mic_frame().tobytes()


def _drain_loop():
    while not _drain_stop.is_set():
        if _mic_stream is None:
            time.sleep(0.05)
            continue
        try:
            _read_mic_frame()
        except Exception as e:
            logger.warning("麦克风 drain 异常: %s", e)
            break


def _start_drain_thread():
    global _drain_thread
    if _drain_thread is not None and _drain_thread.is_alive():
        return
    _drain_stop.clear()
    _drain_thread = threading.Thread(target=_drain_loop, name="mic-drain", daemon=True)
    _drain_thread.start()


def _stop_drain_thread():
    _drain_stop.set()
    global _drain_thread
    if _drain_thread is not None:
        _drain_thread.join(timeout=2.0)
        _drain_thread = None


def _write_wav(path: str, pcm: bytes, sample_rate: int = MIC_SAMPLE_RATE):
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(MIC_SAMPLE_WIDTH)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)


def record_audio():
    if _mic_stream is None:
        logger.error("麦克风流未打开")
        return None

    was_draining = _drain_thread is not None and _drain_thread.is_alive()
    if was_draining:
        _stop_drain_thread()

    record_hold_event.clear()
    logger.info("开始录音")

    try:
        frames: list[bytes] = []
        wait_deadline = time.time() + RECORD_START_TIMEOUT
        while time.time() < wait_deadline:
            buffer = _read_mic_bytes()
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

            buffer = _read_mic_bytes()
            frames.append(buffer)
            if _chunk_energy(buffer) >= recognizer.energy_threshold:
                last_speech_time = time.time()

        pcm = b"".join(frames)
        samples = np.frombuffer(pcm, dtype=np.int16)
        if samples.size == 0:
            logger.warning("录音为空，跳过本轮")
            return None
        rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
        if rms < MIN_RECORD_RMS:
            logger.warning("录音音量过低 (rms=%.0f)，跳过本轮", rms)
            return None

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            wav_path = tf.name
        _write_wav(wav_path, pcm)
        return wav_path
    except Exception as e:
        logger.error("录音失败: %s", e)
        return None
    finally:
        if was_draining and not _wake_listen_enabled:
            _start_drain_thread()


def calibrate_noise(model_path: str | os.PathLike):
    logger.info("正在校准环境噪音，请保持安静 1 秒...")
    if not _ensure_wake_audio(model_path):
        logger.error("麦克风初始化失败")
        return
    try:
        energies: list[float] = []
        deadline = time.time() + 1.0
        while time.time() < deadline:
            energies.append(_chunk_energy(_read_mic_bytes()))
        if energies:
            avg = sum(energies) / len(energies)
            recognizer.energy_threshold = min(
                max(300, avg * 1.2), RECORD_ENERGY_THRESHOLD_MAX
            )
        logger.info("噪声校准完成 threshold=%.0f", recognizer.energy_threshold)
    except Exception as e:
        logger.error("麦克风校准失败: %s", e)


def _open_mic_stream(pa: pyaudio.PyAudio):
    global _downsample_factor
    with suppress_native_stderr():
        try:
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=FRAME_SAMPLES,
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
                    frames_per_buffer=FRAME_SAMPLES * 3,
                )
                _downsample_factor = 3
                return stream
            except Exception:
                logger.error("麦克风被占用或无法打开，请检查硬件")
                return None


def _ensure_wake_audio(model_path: str | os.PathLike) -> bool:
    global _wake_model, _pa, _mic_stream, _wake_audio_ready

    if _wake_audio_ready and _mic_stream is not None:
        return True

    model_path = os.fspath(model_path)
    if _wake_model is None:
        logger.info("正在加载唤醒模型...")
        with suppress_native_stderr():
            try:
                _wake_model = Model(
                    wakeword_models=[model_path], inference_framework="onnx"
                )
            except TypeError:
                _wake_model = Model(wakeword_model_paths=[model_path])
        logger.info("唤醒模型已加载")

    if _pa is None:
        with suppress_native_stderr():
            _pa = pyaudio.PyAudio()

    if _mic_stream is None:
        _mic_stream = _open_mic_stream(_pa)
        if _mic_stream is None:
            _wake_audio_ready = False
            return False

    _wake_audio_ready = True
    return True


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

    prepare_wake_listen()
    wake_event.clear()
    hit_streak = 0

    while True:
        if wake_event.is_set():
            logger.info("唤醒 source=button")
            pause_wake_listen()
            wake_display()
            return True

        frame = _read_mic_frame()
        if frame.size != FRAME_SAMPLES:
            continue

        prediction = _wake_model.predict(frame)

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
                    pause_wake_listen()
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
