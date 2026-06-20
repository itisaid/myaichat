import asyncio
import sys
import threading

from config import wake_event

user_text_queue: asyncio.Queue[str] | None = None


def start_text_debug_reader(loop: asyncio.AbstractEventLoop) -> asyncio.Queue[str]:
    global user_text_queue
    user_text_queue = asyncio.Queue()

    def _reader():
        print("\n[TEXT_DEBUG] 输入 /wake 或 Enter 唤醒，唤醒后输入用户话")
        while True:
            try:
                line = sys.stdin.readline()
            except EOFError:
                break
            if not line:
                break
            text = line.rstrip("\n")
            if text == "" or text == "/wake":
                print("[TEXT_DEBUG] 唤醒")
                wake_event.set()
            else:
                asyncio.run_coroutine_threadsafe(user_text_queue.put(text), loop)

    threading.Thread(target=_reader, daemon=True).start()
    return user_text_queue
