import asyncio
import json

import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates

from config import app_state
from speaker_loop import smart_speaker_loop
from websocket_manager import manager

app = FastAPI()
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

    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(smart_speaker_loop(manager))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
