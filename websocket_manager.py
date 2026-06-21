import json

from fastapi import WebSocket

from config import TEXT_DEBUG, app_state
from llm import get_capabilities
from log_config import get_logger

logger = get_logger("ws")


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    def _config_payload(self) -> str:
        return json.dumps(
            {
                "type": "config_update",
                "model": app_state["model"],
                "enable_thinking": app_state["enable_thinking"],
                "enable_search": app_state["enable_search"],
                "capabilities": get_capabilities(app_state["model"]),
            }
        )

    def _status_payload(self) -> str:
        phase = app_state["phase"]
        return json.dumps(
            {
                "type": "status",
                "text": app_state["status_text"],
                "phase": phase,
                "wake_enabled": app_state["wake_enabled"],
                "record_hold_enabled": phase == "listening" and not TEXT_DEBUG,
                "stop_enabled": app_state["stop_enabled"],
            }
        )

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info("客户端连接 (共 %d)", len(self.active_connections))
        await websocket.send_text(self._config_payload())
        await websocket.send_text(self._status_payload())

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info("客户端断开 (共 %d)", len(self.active_connections))

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                pass

    async def broadcast_config(self):
        await self.broadcast(self._config_payload())

    async def broadcast_status(
        self, text: str, phase: str, *, stop_enabled: bool | None = None
    ):
        app_state["status_text"] = text
        app_state["phase"] = phase
        app_state["wake_enabled"] = phase == "sleeping"
        if stop_enabled is not None:
            app_state["stop_enabled"] = stop_enabled
        elif phase not in ("transcribing", "speaking"):
            app_state["stop_enabled"] = False
        await self.broadcast(self._status_payload())


manager = ConnectionManager()
