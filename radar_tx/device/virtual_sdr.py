# -*- coding: utf-8 -*-
"""虚拟 SDR 后端：无硬件模式。

接收 IQ 后立即交给回环解码器验证，并缓存最近一次解码结果供 Web 诊断显示；
同时模拟真机供样节奏（按采样率等待），保证调度时序与真机一致。
"""

from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np

from .base import TxBackend


class VirtualSdr(TxBackend):
    def __init__(self, sample_rate: int, on_iq: Optional[Callable[[np.ndarray], None]] = None,
                 underflow_gap_ms: float = 15.0, realtime: bool = True) -> None:
        super().__init__(underflow_gap_ms=underflow_gap_ms)
        self._sample_rate = sample_rate
        self._on_iq = on_iq
        self._realtime = realtime
        self.stats.connected = True

    def _apply_config(self, freq_hz: int, bandwidth_hz: int, attenuation_db: float, sample_rate: int) -> None:
        self._sample_rate = sample_rate

    def _push(self, iq: np.ndarray) -> None:
        if self._on_iq is not None:
            self._on_iq(iq)
        if self._realtime and len(iq) > 0:
            # 按真实采样率耗时占位，模拟硬件供样节奏
            time.sleep(len(iq) / self._sample_rate)

    def _apply_mute(self) -> None:
        pass
