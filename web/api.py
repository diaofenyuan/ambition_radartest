# -*- coding: utf-8 -*-
"""Web API：REST 控制接口 + WebSocket 状态推送。

所有控制接口只调用 TxService 的状态方法，业务校验集中在服务层；
接口层负责参数解析、错误到 HTTP 状态码的映射和静态页服务。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from radar_tx.scheduler.tx_service import TxService

STATIC_DIR = Path(__file__).resolve().parent / "static"


class SafetyBody(BaseModel):
    confirmed: bool


class TeamBody(BaseModel):
    team: str


class AutoVaryBody(BaseModel):
    enabled: bool


class JammerBody(BaseModel):
    level: int | None = Field(None, ge=1, le=3)
    key: str | None = None


class CommandBody(BaseModel):
    enabled: bool | None = None
    payload_hex: str | None = None


class AttenuationBody(BaseModel):
    wave: str
    attenuation_db: float


def create_app(service: TxService) -> FastAPI:
    app = FastAPI(title="雷达无线电自测模拟发射系统", on_shutdown=[service.shutdown])

    def _call(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.get("/")
    def index():
        # 强制浏览器每次回源校验：静态页改动后必须立即生效，避免用户看到旧版界面
        return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/api/state")
    def state():
        return service.snapshot()

    @app.post("/api/safety")
    def safety(body: SafetyBody):
        service.set_safety(body.confirmed)
        return {"ok": True}

    @app.post("/api/team")
    def team(body: TeamBody):
        _call(service.set_team, body.team)
        return {"ok": True}

    @app.post("/api/auto_vary")
    def auto_vary(body: AutoVaryBody):
        service.set_auto_vary(body.enabled)
        return {"ok": True}

    @app.post("/api/info/start")
    def info_start():
        _call(service.start, "info")
        return {"ok": True}

    @app.post("/api/info/stop")
    def info_stop():
        service.stop("info")
        return {"ok": True}

    @app.post("/api/jammer/start")
    def jammer_start():
        _call(service.start, "jammer")
        return {"ok": True}

    @app.post("/api/jammer/stop")
    def jammer_stop():
        service.stop("jammer")
        return {"ok": True}

    @app.post("/api/jammer/params")
    def jammer_params(body: JammerBody):
        _call(service.set_jammer, level=body.level, key=body.key)
        return {"ok": True}

    @app.post("/api/command/{cmd_id}")
    def set_command(cmd_id: str, body: CommandBody):
        try:
            cid = int(cmd_id, 16)
        except ValueError:
            raise HTTPException(status_code=400, detail="命令号必须为十六进制，如 0x0A01")
        _call(service.set_command, cid, enabled=body.enabled, payload_hex=body.payload_hex)
        return {"ok": True}

    @app.post("/api/command/{cmd_id}/random")
    def random_command(cmd_id: str):
        try:
            cid = int(cmd_id, 16)
        except ValueError:
            raise HTTPException(status_code=400, detail="命令号必须为十六进制")
        _call(service.randomize_command, cid)
        return {"ok": True}

    @app.post("/api/attenuation")
    def attenuation(body: AttenuationBody):
        if body.wave not in ("info", "jammer"):
            raise HTTPException(status_code=400, detail="wave 必须为 info / jammer")
        _call(service.set_attenuation, body.wave, body.attenuation_db)
        return {"ok": True}

    @app.post("/api/mute")
    def mute():
        service.mute_all()
        return {"ok": True}

    @app.websocket("/ws")
    async def ws_push(websocket: WebSocket):
        """每 0.5 s 推送一次状态快照，界面无需轮询。"""
        await websocket.accept()
        try:
            while True:
                await websocket.send_text(json.dumps(service.snapshot(), ensure_ascii=False))
                await asyncio.sleep(0.5)
        except (WebSocketDisconnect, RuntimeError, ConnectionError):
            # 客户端直接关页/断网时，send 可能抛 RuntimeError/ConnectionError 而不是
            # WebSocketDisconnect；推送循环直接结束即可，不必让异常冒泡刷栈
            pass

    return app
