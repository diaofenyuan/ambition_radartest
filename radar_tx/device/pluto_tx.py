# -*- coding: utf-8 -*-
"""PlutoSDR（AD9363）真机发射后端，基于 pyadi-iio。

要点：
- 双板必须分子网（ip:192.168.2.1 / ip:192.168.3.1），URI 用稳定的 ip: 形式；
- 非循环缓冲连续推流，tx_hardwaregain 以负值表示衰减；
- Pluto 的非循环 TX 缓冲建立后长度不可变，而信息波每 3 个周期会出现一次
  10 包超长周期；因此后端必须按固定块推送并跨周期保留余量，否则会触发
  "Cannot change buffer length on the fly" 并终止发射线程；
- pyadi-iio 延迟导入，未安装或无硬件时不影响仿真模式运行。
"""

from __future__ import annotations

import numpy as np

from .base import TxBackend


class PlutoTx(TxBackend):
    def __init__(self, uri: str, underflow_gap_ms: float = 15.0,
                 block_samples: int = 0) -> None:
        super().__init__(underflow_gap_ms=underflow_gap_ms)
        self._uri = uri
        self._sdr = None
        # 0 表示不启用定长分块（保留给直接调用后端的场景）；服务层始终传入周期采样数
        self._block_samples = int(block_samples)
        self._pending = np.zeros(0, dtype=np.complex64)

    def _ensure_device(self):
        if self._sdr is not None:
            return self._sdr
        # 真机模式必须先定位 libiio 原生库：缺失时给出可读原因，
        # 而不是让 pylibiio 在导入阶段抛一句 TypeError。
        from .libiio_runtime import ensure_libiio_available
        libiio_ok, detail = ensure_libiio_available()
        if not libiio_ok:
            raise RuntimeError(detail)
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
        data = np.ascontiguousarray(iq, dtype=np.complex64)
        if self._block_samples <= 0:
            sdr.tx(data)
            return

        # 每次调用只推一个定长块，保持设备消耗速率恒定：
        # 超长周期的尾部留到下次，短周期补零；不能在单次调用里把余量全部排空，
        # 否则会一次性推入两倍采样、破坏 10Hz 节拍。
        if self._pending.size:
            data = np.concatenate((self._pending, data))
        take = min(len(data), self._block_samples)
        block = np.zeros(self._block_samples, dtype=np.complex64)
        block[:take] = data[:take]
        self._pending = data[take:].copy()
        sdr.tx(block)

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
        self._pending = np.zeros(0, dtype=np.complex64)
