# -*- coding: utf-8 -*-
"""界面退出检测：覆盖刷新、多页面及失联回收。"""

import asyncio
from types import SimpleNamespace

import pytest

from web.api import create_app
from web.ui_session import UiSessions


@pytest.fixture
def sessions():
    now = [0.0]
    return UiSessions(clock=lambda: now[0]), now


def test_waits_for_first_page(sessions):
    ui, now = sessions
    now[0] = 1000
    assert not ui.should_exit()


def test_last_page_closes_after_grace(sessions):
    ui, now = sessions
    ui.touch("page")
    ui.discard("page", closed=True)
    now[0] = 2.9
    assert not ui.should_exit()
    now[0] = 3
    assert ui.should_exit()


def test_other_page_keeps_backend_alive(sessions):
    ui, now = sessions
    ui.touch("first")
    ui.touch("second")
    ui.discard("first", closed=True)
    now[0] = 10
    assert not ui.should_exit()
    ui.discard("second", closed=True)
    now[0] = 13
    assert ui.should_exit()


def test_refresh_or_reconnect_cancels_exit(sessions):
    ui, now = sessions
    ui.touch("old")
    ui.discard("old", closed=True)
    now[0] = 2
    ui.touch("new")
    now[0] = 5
    assert not ui.should_exit()
    ui.discard("new")
    now[0] = 7
    ui.touch("new")
    now[0] = 10
    assert not ui.should_exit()


def test_inflight_request_cannot_revive_closed_page(sessions):
    ui, now = sessions
    ui.touch("page")
    ui.discard("page", closed=True)
    now[0] = 2
    ui.touch("page")
    ui.discard("page")
    now[0] = 3
    assert ui.should_exit()


def test_lost_polling_page_expires(sessions):
    ui, now = sessions
    ui.touch("page")
    now[0] = 60
    assert not ui.should_exit()
    now[0] = 90
    assert not ui.should_exit()
    now[0] = 93
    assert ui.should_exit()


@pytest.mark.parametrize("mute_fails", [False, True])
def test_auto_exit_mutes_before_exit_and_always_releases_devices(monkeypatch, mute_fails):
    monkeypatch.setattr(UiSessions, "should_exit", lambda self: True)
    events = []

    async def run():
        exited = asyncio.Event()

        def mute():
            events.append("mute")
            if mute_fails:
                raise RuntimeError("模拟设备静音失败")

        def request_exit():
            events.append("exit")
            exited.set()

        service = SimpleNamespace(mute_all=mute, shutdown=lambda: events.append("close"))
        app = create_app(service, on_ui_closed=request_exit)
        async with app.router.lifespan_context(app):
            await asyncio.wait_for(exited.wait(), timeout=2)
            assert events == ["mute", "exit"]

    asyncio.run(run())
    assert events == ["mute", "exit", "close"]
