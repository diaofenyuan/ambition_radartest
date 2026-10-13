# -*- coding: utf-8 -*-
"""发射后端统一接口。

调度层只依赖本接口，仿真与真机后端可互换；
统计口径统一（供样/欠采/采样数），保证 Web 诊断面板在两种模式下行为一致。
欠采判定：推流周期超过上一包信号时长 + 余量，说明设备端供样断流，
射频输出会出现波形断点——这是发射端测试最常见的隐性故障，单独计数。
"""

from __future__ import annotations

import abc
import threading
import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class TxStats:
    """发射运行统计。"""
    samples_sent: int = 0
    pushes: int = 0
    underflow_count: int = 0
    stream_margin_ms: float = 0.0   # 供样余量：信号时长 - 推流周期，负值表示断流风险
    connected: bool = False
    freq_hz: int = 0
    bandwidth_hz: int = 0
    attenuation_db: float = 0.0
    extra: dict = field(default_factory=dict)


class TxBackend(abc.ABC):
    """发射后端抽象：配置射频参数 + 推送 IQ 流。"""

    def __init__(self, underflow_gap_ms: float = 15.0) -> None:
        self.stats = TxStats()
        self._lock = threading.Lock()
        self._last_push_ts: float | None = None
        self._last_duration = 0.0
        self._sample_rate = 1_000_000
        self._underflow_gap_s = underflow_gap_ms / 1000.0

    def configure(self, freq_hz: int, bandwidth_hz: int, attenuation_db: float, sample_rate: int) -> None:
        self._sample_rate = sample_rate
        with self._lock:
            changed = (self.stats.freq_hz != freq_hz or self.stats.bandwidth_hz != bandwidth_hz
                       or self.stats.attenuation_db != attenuation_db)
            self.stats.freq_hz = freq_hz
            self.stats.bandwidth_hz = bandwidth_hz
            self.stats.attenuation_db = attenuation_db
        if changed or not self.stats.connected:
            self._apply_config(freq_hz, bandwidth_hz, attenuation_db, sample_rate)

    def transmit(self, iq: np.ndarray) -> None:
        """推送一段基带 IQ，并按供样余量做欠采告警。"""
        now = time.monotonic()
        duration = len(iq) / self._sample_rate if self._sample_rate else 0.0
        with self._lock:
            if self._last_push_ts is not None and self._last_duration > 0:
                period = now - self._last_push_ts
                self.stats.stream_margin_ms = (self._last_duration - period) * 1000.0
                if period > self._last_duration + self._underflow_gap_s:
                    self.stats.underflow_count += 1
        self._last_push_ts = now
        self._last_duration = duration
        self._push(iq)
        with self._lock:
            self.stats.samples_sent += len(iq)
            self.stats.pushes += 1

    def mute(self) -> None:
        """硬静音：衰减拉满（实现层可同时关本振）。"""
        with self._lock:
            self.stats.attenuation_db = 89.0
        self._apply_mute()

    def close(self) -> None:
        self._close()
        with self._lock:
            self.stats.connected = False

    # ---- 子类实现 ----

    @abc.abstractmethod
    def _apply_config(self, freq_hz: int, bandwidth_hz: int, attenuation_db: float, sample_rate: int) -> None: ...

    @abc.abstractmethod
    def _push(self, iq: np.ndarray) -> None: ...

    @abc.abstractmethod
    def _apply_mute(self) -> None: ...

    def _close(self) -> None:  # 默认空实现
        pass
