import os
import select
import subprocess
import sys
import threading
import time

from config import (
    DISPLAY_WAKE_ENABLED,
    DISPLAY_WAKE_USE_XDOTOOL,
    TOUCH_MONITOR_ENABLED,
)
from log_config import get_logger

logger = get_logger("display")

_wake_lock = threading.Lock()
_last_wake_at = 0.0
_DEBOUNCE_SECONDS = 0.2


def _display_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("DISPLAY", ":0")
    xauth = os.path.expanduser("~/.Xauthority")
    if os.path.isfile(xauth):
        env.setdefault("XAUTHORITY", xauth)
    return env


def _run_cmd(args: list[str]) -> bool:
    try:
        result = subprocess.run(
            args,
            env=_display_env(),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode != 0:
            err = (result.stderr or result.stdout or "").strip()
            if err:
                logger.warning("%s failed: %s", " ".join(args), err)
            return False
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("%s error: %s", " ".join(args), exc)
        return False


def wake_display() -> None:
    """Turn on HDMI/DPMS display. No-op on non-Linux or when disabled."""
    if not DISPLAY_WAKE_ENABLED or sys.platform != "linux":
        return

    global _last_wake_at
    now = time.monotonic()
    with _wake_lock:
        if now - _last_wake_at < _DEBOUNCE_SECONDS:
            return
        _last_wake_at = now

    xset_ok = _run_cmd(["xset", "dpms", "force", "on"])
    _run_cmd(["vcgencmd", "display_power", "1"])
    if not xset_ok and DISPLAY_WAKE_USE_XDOTOOL:
        _run_cmd(["xdotool", "mousemove_relative", "--", "1", "0"])


def _touch_name_hints() -> tuple[str, ...]:
    return ("touch", "ctouch", "touchscreen", "fusion", "eeti", "goodix")


def _is_touch_device(device) -> bool:
    from evdev import ecodes

    caps = device.capabilities()
    if ecodes.EV_ABS not in caps:
        return False

    name = (device.name or "").lower()
    if any(hint in name for hint in _touch_name_hints()):
        return True

    abs_codes = {code for code, _ in caps.get(ecodes.EV_ABS, [])}
    has_xy = ecodes.ABS_X in abs_codes and ecodes.ABS_Y in abs_codes
    has_mt = ecodes.ABS_MT_POSITION_X in abs_codes or ecodes.ABS_MT_SLOT in abs_codes
    if has_mt:
        return True
    if has_xy and ecodes.EV_REL not in caps:
        return True
    return False


def _find_touch_devices():
    try:
        import evdev
    except ImportError:
        logger.warning("evdev 未安装，触摸监听已禁用")
        return []

    devices = []
    for path in evdev.list_devices():
        try:
            device = evdev.InputDevice(path)
            if _is_touch_device(device):
                devices.append(device)
        except (OSError, PermissionError) as exc:
            logger.debug("skip %s: %s", path, exc)
    return devices


def _touch_event_wakes(event) -> bool:
    from evdev import ecodes

    if event.type == ecodes.EV_KEY:
        return event.code in (ecodes.BTN_TOUCH, ecodes.BTN_LEFT) and event.value
    if event.type == ecodes.EV_ABS:
        return event.code in (
            ecodes.ABS_X,
            ecodes.ABS_Y,
            ecodes.ABS_MT_POSITION_X,
            ecodes.ABS_MT_POSITION_Y,
            ecodes.ABS_MT_TRACKING_ID,
        )
    return False


def _touch_monitor_loop() -> None:
    devices = _find_touch_devices()
    if not devices:
        logger.warning("未找到触摸输入设备")
        return

    names = ", ".join(device.name or device.path for device in devices)
    logger.info("触摸监听: %s", names)

    while True:
        try:
            ready, _, _ = select.select(devices, [], [], 1.0)
            for device in ready:
                for event in device.read():
                    if _touch_event_wakes(event):
                        wake_display()
        except OSError as exc:
            logger.warning("触摸监听异常: %s", exc)
            time.sleep(5)
            devices = _find_touch_devices()
            if not devices:
                time.sleep(10)


def start_touch_monitor() -> None:
    if not TOUCH_MONITOR_ENABLED or not DISPLAY_WAKE_ENABLED:
        return
    if sys.platform != "linux":
        return

    thread = threading.Thread(
        target=_touch_monitor_loop,
        daemon=True,
        name="touch-monitor",
    )
    thread.start()
