# -*- coding: utf-8 -*-
"""界面会话存活检测，仅由 API 事件循环访问。"""

from __future__ import annotations

import time
from collections.abc import Callable


class UiSessions:
    def __init__(self, grace_seconds: float = 3.0, timeout_seconds: float = 90.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._grace = grace_seconds
        self._timeout = timeout_seconds
        self._clock = clock
        self._sessions: dict[str, float] = {}
        self._closed: set[str] = set()
        self._seen = False
        self._empty_since: float | None = None

    def touch(self, session_id: str) -> None:
        # 关页后可能仍有在途轮询或推送，不能让这些请求重新激活已关闭的页面。
        if session_id in self._closed:
            return
        self._seen = True
        self._sessions[session_id] = self._clock()
        self._empty_since = None

    def discard(self, session_id: str, *, closed: bool = False) -> None:
        if closed:
            self._closed.add(session_id)
        self._sessions.pop(session_id, None)
        if self._seen and not self._sessions and self._empty_since is None:
            self._empty_since = self._clock()

    def should_exit(self) -> bool:
        now = self._clock()
        for session_id, last_seen in list(self._sessions.items()):
            if now - last_seen >= self._timeout:
                self.discard(session_id)
        # 启动后等待首次打开页面；最后一页离开后留出刷新、重连的宽限期。
        return (self._seen and self._empty_since is not None
                and now - self._empty_since >= self._grace)
