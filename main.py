import asyncio
import json

from log_config import get_logger, setup_logging

setup_logging()

import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from config import LOG_LEVEL, TEXT_DEBUG, app_state, cancel_event, record_hold_event, wake_event
from display import start_touch_monitor, wake_display
from speaker_loop import smart_speaker_loop
from text_debug import start_text_debug_reader
from websocket_manager import manager

logger = get_logger("startup")

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse(
        request=request, name="index.html", context={"request": request}
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if message.get("type") == "change_model":
                app_state["model"] = message.get("model")
                logger.info("模型切换为 %s", app_state["model"])
                await manager.broadcast_config()

            elif message.get("type") == "set_options":
                if "enable_thinking" in message:
                    app_state["enable_thinking"] = bool(message["enable_thinking"])
                if "enable_search" in message:
                    app_state["enable_search"] = bool(message["enable_search"])
                await manager.broadcast_config()
                logger.info(
                    "选项更新 thinking=%s search=%s",
                    app_state["enable_thinking"],
                    app_state["enable_search"],
                )

            elif message.get("type") == "wake":
                wake_display()
                wake_event.set()
                logger.info("前端按钮触发唤醒")

            elif message.get("type") == "display_wake":
                wake_display()

            elif message.get("type") == "record_hold":
                wake_display()
                if app_state["phase"] != "listening":
                    continue
                if message.get("active"):
                    record_hold_event.set()
                else:
                    record_hold_event.clear()

            elif message.get("type") == "stop":
                wake_display()
                if app_state.get("stop_enabled"):
                    cancel_event.set()
                    logger.info("前端按钮触发终止")

    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.on_event("startup")
async def startup_event():
    start_touch_monitor()
    if TEXT_DEBUG:
        start_text_debug_reader(asyncio.get_running_loop())
        logger.info("TEXT_DEBUG 模式：终端 /wake + 文本输入，浏览器仅观察")
    asyncio.create_task(smart_speaker_loop(manager))


if __name__ == "__main__":
    logger.info("服务启动 http://0.0.0.0:8000")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level=LOG_LEVEL.lower(),
        access_log=TEXT_DEBUG,
    )
