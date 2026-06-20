import asyncio
import json

import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from config import TEXT_DEBUG, app_state, record_hold_event, wake_event
from speaker_loop import smart_speaker_loop
from text_debug import start_text_debug_reader
from websocket_manager import manager

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
                print(f"前端已将模型切换为: {app_state['model']}")
                await manager.broadcast_config()

            elif message.get("type") == "set_options":
                if "enable_thinking" in message:
                    app_state["enable_thinking"] = bool(message["enable_thinking"])
                if "enable_search" in message:
                    app_state["enable_search"] = bool(message["enable_search"])
                await manager.broadcast_config()
                print(
                    f"选项更新: 深度思考={app_state['enable_thinking']}, "
                    f"联网搜索={app_state['enable_search']}"
                )

            elif message.get("type") == "wake":
                wake_event.set()
                print("前端按钮触发唤醒")

            elif message.get("type") == "record_hold":
                if app_state["phase"] != "listening":
                    continue
                if message.get("active"):
                    record_hold_event.set()
                else:
                    record_hold_event.clear()

    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.on_event("startup")
async def startup_event():
    if TEXT_DEBUG:
        start_text_debug_reader(asyncio.get_running_loop())
        print("TEXT_DEBUG 模式：终端 /wake + 文本输入，浏览器仅观察")
    asyncio.create_task(smart_speaker_loop(manager))


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="warning" if TEXT_DEBUG else "info",
        access_log=not TEXT_DEBUG,
    )
