# -*- coding: utf-8 -*-
"""PlutoSDR（AD9363）真机发射后端，基于 pyadi-iio。

要点：
- 双板必须分子网（ip:192.168.2.1 / ip:192.168.3.1），URI 用稳定的 ip: 形式；
- TX 硬件增益以负值表示衰减，属性名随 pyadi 版本变化（见 _set_tx_gain）；
- Pluto 的非循环 TX 缓冲建立后长度不可变，而信息波每 3 个周期会出现一次
  10 包超长周期；因此后端必须按固定块推送并跨周期保留余量，否则会触发
  "Cannot change buffer length on the fly" 并终止发射线程；
- 送入 sdr.tx() 的数据必须已换算到 DAC 的 int16 满量程，见 to_dac_scale；
- pyadi-iio 延迟导入，未安装或无硬件时不影响仿真模式运行。
"""

from __future__ import annotations

import numpy as np

from .base import TxBackend

# pyadi 的 tx() 对复数样点只做 astype(int16)，不做任何归一化：
# 传 ±1 的浮点会被直接截断成 0，DAC 输出静默——实测基带单音比 LO 泄漏
# 还低 12 dB，等于完全没发（同一块板改用满量程后提升 43.5 dB）。
# 波形层统一保持 ±1 归一化（仿真回环、离线分析都依赖它），只在这一层换算。
DAC_FULL_SCALE = 32767.0


def to_dac_scale(iq: np.ndarray) -> np.ndarray:
    """把 ±1 归一化的 IQ 换算到 DAC 的 int16 满量程。

    所有真实发射路径都必须经过这里：驱动后端、独立探针、链路体检。
    漏掉任何一处，那条路径就是"界面显示在发、频谱上什么都没有"。
    """
    return np.ascontiguousarray(iq, dtype=np.complex64) * DAC_FULL_SCALE


# AD9361 的 TX 模拟滤波器带宽贴着信号带宽设会被严重削顶：jammer_3 的
# 250 kHz 实测比 540 kHz 低 21 dB，端到端一帧都解不出；同一套功率参数下
# 把带宽放到 540 kHz 就能稳定解调。所以这里设下限，而不是照抄信号带宽。
TX_RF_BANDWIDTH_MIN_HZ = 540_000


def tx_rf_bandwidth(bandwidth_hz: int, sample_rate: int) -> int:
    """TX 模拟滤波带宽：不低于下限，也不超过采样率（再宽只是多放噪声）。"""
    return int(min(max(int(bandwidth_hz), TX_RF_BANDWIDTH_MIN_HZ), int(sample_rate)))


def _tx_gain_attr(sdr) -> str:
    """返回当前 pyadi 版本里可写的 TX 增益属性名。

    属性名不稳定：pyadi 0.0.21 是 tx_hardwaregain_chan0，另一些版本是
    tx_hardwaregain。写错名字不会报错——普通对象会把赋值静静存成一个实例
    属性，硬件增益原地不动，表现是"能发但功率不可调、MUTE 也不静音"。
    所以必须按优先级探测，且 chan0 要排在前面（防止被错误赋值过的同名属性骗到）。
    """
    for name in ("tx_hardwaregain_chan0", "tx_hardwaregain"):
        if hasattr(sdr, name):
            return name
    return ""


def _set_tx_gain(sdr, attenuation_db: float) -> None:
    name = _tx_gain_attr(sdr)
    if not name:
        raise RuntimeError(
            "当前 pyadi-iio 版本上没有可写的 TX 硬件增益属性，无法设置发射功率")
    setattr(sdr, name, -float(attenuation_db))


# USB 掉线的典型报错：libiio 抛 OSError（110 host unreachable / 19 no such device），
# 或者 pyadi 在重连失败时给 "No device found"。其它异常（参数非法、缓冲长度变更）
# 属于配置问题，重连没有意义，必须原样上抛。
_DISCONNECT_HINTS = ("host unreachable", "no device found", "no such device",
                     "resource busy", "errno 110", "errno 19")


def _is_disconnect(exc: Exception) -> bool:
    """判断异常是否属于"设备掉线"，只有这类才值得重连重试。"""
    text = f"{type(exc).__name__}: {exc}".lower()
    return isinstance(exc, OSError) or any(h in text for h in _DISCONNECT_HINTS)


def resolve_uri(uri_spec: str) -> str:
    """把 ``serial:`` 前缀解析成当前枚举到的 URI，其余形式原样返回。

    USB 重连会换端口号（usb:2.13.5 → usb:2.15.5），配置里写死 usb: 地址第二天
    就可能打不开；按 serial 绑定则不受端口号变化影响。
    """
    if not uri_spec.lower().startswith("serial:"):
        return uri_spec
    want = uri_spec.split(":", 1)[1].strip()
    if not want:
        raise RuntimeError("serial: 绑定缺少 serial 号")
    from .discovery import scan_pluto_devices
    for dev in scan_pluto_devices():
        if dev.serial.startswith(want):
            return dev.uri
    raise RuntimeError(f"没有找到 serial 以 {want} 开头的 PlutoSDR")


class PlutoTx(TxBackend):
    def __init__(self, uri: str, underflow_gap_ms: float = 15.0,
                 block_samples: int = 0) -> None:
        super().__init__(underflow_gap_ms=underflow_gap_ms)
        self._uri = uri
        # 记住 serial 用于掉线后重新定位：USB 重连会把端口号换掉
        # （实测 usb:2.13.5 → usb:2.15.5 → usb:2.6.5），只认 URI 就再也找不到板子。
        self._serial = ""
        self._last_config: tuple | None = None
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
        # serial 必须在打开设备之前定下来：一旦这台上下文被占用，
        # 再扫描就看不到它，回查只会得到空串，掉线自愈也就失效了。
        # serial: 形式的前缀本身就是 serial 的一部分，直接用。
        if self._uri.lower().startswith("serial:") and not self._serial:
            self._serial = self._uri.split(":", 1)[1].strip()
        resolved = resolve_uri(self._uri)
        if not self._serial:
            try:
                from .discovery import scan_pluto_devices
                for dev in scan_pluto_devices():
                    if dev.uri == resolved:
                        self._serial = dev.serial
                        break
            except Exception:
                pass
        import adi  # 延迟导入：仿真模式不需要该依赖
        sdr = adi.Pluto(resolved)
        sdr.tx_cyclic_buffer = False
        self._uri = resolved  # 后续复用解析结果，不必每次扫描
        self._sdr = sdr
        self.stats.connected = True
        # 设备复位会丢掉全部射频设置，重连后必须把上一次的配置补回去，
        # 否则会按默认频点/增益发射，看起来"连上了但参数全不对"。
        if self._last_config is not None:
            freq, bw, atten, rate = self._last_config
            sdr.sample_rate = int(rate)
            sdr.tx_lo = int(freq)
            sdr.tx_rf_bandwidth = tx_rf_bandwidth(bw, rate)
            _set_tx_gain(sdr, atten)
        return sdr

    def _reconnect(self, reason: Exception) -> None:
        """设备掉线后按 serial 重新定位并重连。"""
        with self._lock:
            self.stats.connected = False
            self.stats.extra["reconnects"] = self.stats.extra.get("reconnects", 0) + 1
            self.stats.extra["last_error"] = f"{type(reason).__name__}: {reason}"
        self._sdr = None
        # 设备端缓冲随掉线一起消失，手里这份余量已经没有归属
        self._pending = np.zeros(0, dtype=np.complex64)
        if self._serial:
            try:
                from .discovery import scan_pluto_devices
                for dev in scan_pluto_devices():
                    if dev.serial == self._serial:
                        self._uri = dev.uri
                        break
            except Exception:
                pass  # 扫描失败就沿用旧 URI 再试一次，由调用方决定是否停发
        self._ensure_device()

    def _apply_config(self, freq_hz: int, bandwidth_hz: int, attenuation_db: float, sample_rate: int) -> None:
        # 先记下配置：设备随时可能掉线，重连后要靠这份记录把射频参数补回去
        self._last_config = (int(freq_hz), int(bandwidth_hz), float(attenuation_db), int(sample_rate))
        sdr = self._ensure_device()
        # 逐项比对后再写：写 sample_rate 会触发 AD9361 时钟重算，实测固定 500 ms，
        # 期间调用方（发射线程）被锁住、推流断流。衰减是拖动时的高频操作，
        # 不能每次都陪跑一次时钟重算——只在值真的变化时才写。
        bw = tx_rf_bandwidth(bandwidth_hz, sample_rate)
        if int(sdr.sample_rate) != int(sample_rate):
            sdr.sample_rate = int(sample_rate)
        if int(sdr.tx_lo) != int(freq_hz):
            sdr.tx_lo = int(freq_hz)
        if int(sdr.tx_rf_bandwidth) != bw:
            sdr.tx_rf_bandwidth = bw
        # AD9361 发射增益域：0 dB 为最大功率，负值等效衰减
        _set_tx_gain(sdr, attenuation_db)

    def _push(self, iq: np.ndarray) -> None:
        try:
            self._push_once(iq)
        except Exception as exc:
            if not _is_disconnect(exc):
                raise
            # 掉线只重连重试一次：连续失败说明板子没回来，应交回调度层停发并报错；
            # 在这里循环重试会把发射线程卡死，界面上也看不到真正原因。
            self._reconnect(exc)
            self._push_once(iq)

    def _push_once(self, iq: np.ndarray) -> None:
        sdr = self._ensure_device()
        data = to_dac_scale(iq)
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
            _set_tx_gain(self._sdr, 89.0)

    def _close(self) -> None:
        if self._sdr is not None:
            # 先静音再拆缓冲：实测 tx_destroy_buffer() 之后射频仍会按最后一帧
            # 继续输出（接收端还能收到 +93 dB 的残留载波），只拆缓冲不算停发。
            try:
                self._apply_mute()
            except Exception:
                pass
            try:
                self._sdr.tx_destroy_buffer()
            except Exception:
                pass
            self._sdr = None
        self._pending = np.zeros(0, dtype=np.complex64)
