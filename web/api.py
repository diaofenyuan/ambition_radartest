# -*- coding: utf-8 -*-
"""Web API：REST 控制接口 + WebSocket 状态推送。

所有控制接口只调用 TxService 的状态方法，业务校验集中在服务层；
接口层负责参数解析、错误到 HTTP 状态码的映射和静态页服务。
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from radar_tx.scheduler.tx_service import TxService
from .ui_session import UiSessions

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


class DeviceRoleBody(BaseModel):
    uri: str = Field(..., min_length=1, max_length=256)
    role: str | None = None


def create_app(service: TxService, on_ui_closed: Callable[[], None] | None = None) -> FastAPI:
    sessions = UiSessions()

    async def watch_ui():
        while True:
            await asyncio.sleep(0.5)
            if sessions.should_exit():
                print("所有界面已关闭，正在静音并退出后端……", flush=True)
                try:
                    await asyncio.to_thread(service.mute_all)
                except Exception:
                    logging.exception("退出前执行 MUTE 失败，继续释放设备")
                finally:
                    on_ui_closed()
                return

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        watcher = asyncio.create_task(watch_ui()) if on_ui_closed is not None else None
        try:
            yield
        finally:
            if watcher is not None:
                watcher.cancel()
                await asyncio.gather(watcher, return_exceptions=True)
            await asyncio.to_thread(service.shutdown)

    app = FastAPI(title="雷达无线电自测模拟发射系统", lifespan=lifespan)

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
    async def state(ui_session: str | None = Query(None, min_length=1, max_length=64)):
        if ui_session is not None:
            sessions.touch(ui_session)
        return service.snapshot()

    @app.post("/api/ui/{session_id}/close")
    async def close_ui(session_id: str):
        sessions.discard(session_id, closed=True)
        return {"ok": True}

    @app.post("/api/devices/scan")
    def devices_scan():
        return {"devices": service.scan_devices()}

    @app.post("/api/devices/assign")
    def devices_assign(body: DeviceRoleBody):
        _call(service.set_device_role, body.uri, body.role)
        return {"ok": True}

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
    async def ws_push(websocket: WebSocket,
                      ui_session: str | None = Query(None, min_length=1, max_length=64)):
        """每 0.5 s 推送一次状态快照，界面无需轮询。"""
        await websocket.accept()
        try:
            while True:
                if ui_session is not None:
                    sessions.touch(ui_session)
                await websocket.send_text(json.dumps(service.snapshot(), ensure_ascii=False))
                await asyncio.sleep(0.5)
        except (WebSocketDisconnect, RuntimeError, ConnectionError):
            # 客户端直接关页/断网时，send 可能抛 RuntimeError/ConnectionError 而不是
            # WebSocketDisconnect；推送循环直接结束即可，不必让异常冒泡刷栈
            pass
        finally:
            if ui_session is not None:
                sessions.discard(ui_session)

    return app
