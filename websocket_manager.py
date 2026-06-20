import json

from fastapi import WebSocket

from config import app_state


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    def _status_payload(self) -> str:
        return json.dumps(
            {
                "type": "status",
                "text": app_state["status_text"],
                "phase": app_state["phase"],
                "wake_enabled": app_state["wake_enabled"],
            }
        )

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        await websocket.send_text(
            json.dumps({"type": "config_update", "model": app_state["model"]})
        )
        await websocket.send_text(self._status_payload())

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                pass

    async def broadcast_status(self, text: str, phase: str):
        app_state["status_text"] = text
        app_state["phase"] = phase
        app_state["wake_enabled"] = phase == "sleeping"
        await self.broadcast(self._status_payload())


manager = ConnectionManager()
