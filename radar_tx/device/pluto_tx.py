# -*- coding: utf-8 -*-
"""PlutoSDR（AD9363）真机发射后端，基于 pyadi-iio。

要点：
- 双板必须分子网（ip:192.168.2.1 / ip:192.168.3.1），URI 用稳定的 ip: 形式；
- 非循环缓冲连续推流，tx_hardwaregain 以负值表示衰减；
- pyadi-iio 延迟导入，未安装或无硬件时不影响仿真模式运行。
"""

from __future__ import annotations

import numpy as np

from .base import TxBackend


class PlutoTx(TxBackend):
    def __init__(self, uri: str, underflow_gap_ms: float = 15.0) -> None:
        super().__init__(underflow_gap_ms=underflow_gap_ms)
        self._uri = uri
        self._sdr = None

    def _ensure_device(self):
        if self._sdr is not None:
            return self._sdr
        import adi  # 延迟导入：仿真模式不需要该依赖
        sdr = adi.Pluto(self._uri)
        sdr.tx_cyclic_buffer = False
        self._sdr = sdr
        self.stats.connected = True
        return sdr

    def _apply_config(self, freq_hz: int, bandwidth_hz: int, attenuation_db: float, sample_rate: int) -> None:
        sdr = self._ensure_device()
        sdr.sample_rate = int(sample_rate)
        sdr.tx_lo = int(freq_hz)
        sdr.tx_rf_bandwidth = int(bandwidth_hz)
        # AD9361 发射增益域：0 dB 为最大功率，负值等效衰减
        sdr.tx_hardwaregain = -float(attenuation_db)

    def _push(self, iq: np.ndarray) -> None:
        sdr = self._ensure_device()
        sdr.tx(np.ascontiguousarray(iq, dtype=np.complex64))

    def _apply_mute(self) -> None:
        if self._sdr is not None:
            self._sdr.tx_hardwaregain = -89.0

    def _close(self) -> None:
        if self._sdr is not None:
            try:
                self._sdr.tx_destroy_buffer()
            except Exception:
                pass
            self._sdr = None
