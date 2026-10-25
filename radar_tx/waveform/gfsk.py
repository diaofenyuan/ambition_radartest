# -*- coding: utf-8 -*-
"""2-GFSK 调制与解调（波形层）。

调制链路：比特 → NRZ(±1) → 矩形脉冲成形 → 高斯低通(BT=0.35) → 频率积分 → IQ。
频偏由射频带宽推导（f_dev = BW/2 - 码率），与规则表 5-23 的 sensitivity 公式一致，
换赛季改带宽后波形自动跟随。
解调器仅用于仿真回环自检：相差鉴频 + 全相位搜索选符号相位，
真实接收链路由雷达系统/开源 RX 完成，不在本项目范围。
"""

from __future__ import annotations

import math
from typing import List

import numpy as np

from ..config import ModulationParams, RfProfile


def gaussian_taps(mod: ModulationParams) -> np.ndarray:
    """生成高斯滤波器抽头（单位直流增益）。

    σ（符号周期域）= sqrt(ln2) / (2π·BT)，对齐 GNU Radio firdes.gaussian 定义；
    单位直流增益保证稳态频偏恰好等于设计值。
    """
    sigma_sym = math.sqrt(math.log(2.0)) / (2 * math.pi * mod.GFSK_BT)
    sigma = sigma_sym * mod.SAMPLES_PER_SYMBOL
    half = mod.GAUSSIAN_SPAN_SYMBOLS * mod.SAMPLES_PER_SYMBOL // 2
    x = np.arange(-half, half + 1, dtype=np.float64)
    taps = np.exp(-x * x / (2 * sigma * sigma))
    return taps / taps.sum()


class GfskModulator:
    """字节流 → 复数基带 IQ。

    规则附录 1 规定空口包之间没有保护间隔（"数据到达即发射"），所以发射线程
    必须按**连续流**调制：滤波器尾样本与相位跨数据块保留。若逐包独立调制再首尾
    相接，每个包边界相位归零，瞬时频偏会出现越过 ±Δf 的跳变，且每包首尾各一个
    符号的频偏只有设定值的一半。
    """

    def __init__(self, mod: ModulationParams, profile: RfProfile, amplitude: float = 0.5) -> None:
        self.mod = mod
        self.profile = profile
        self.amplitude = amplitude
        self._taps = gaussian_taps(mod)
        # 群延迟补偿：滤波器线性相位延迟为 (ntaps-1)/2 个采样
        self._delay = (len(self._taps) - 1) // 2
        self._phase_inc = 2 * math.pi * profile.freq_deviation_hz / mod.SAMPLE_RATE_HZ
        self.reset()

    def reset(self) -> None:
        """丢弃跨块状态。

        发射中断期间射频是静的，把上一轮残留的相位与滤波器历史接到新一轮开头，
        会凭空造出一个本不存在的相位跳变，因此每次重新开始发射都要清一次。
        """
        self._history = np.zeros(self._delay, dtype=np.float64)
        self._phase = 0.0

    def modulate_bytes(self, data: bytes) -> np.ndarray:
        """整块调制，不读写跨块状态（单包波形、离线分析用）。"""
        bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
        return self.modulate_bits(bits)

    def modulate_bits(self, bits: np.ndarray) -> np.ndarray:
        """同 modulate_bytes，输入为比特序列；始终从零相位、零历史开始。"""
        return self._emit(np.asarray(bits, dtype=np.uint8), self._fresh_history(), advance=False)

    def modulate_stream(self, data: bytes) -> np.ndarray:
        """连续流调制：保留滤波器尾样本与相位，允许跨调用首尾相接。"""
        if len(data) == 0:
            return np.zeros(0, dtype=np.complex64)
        bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
        return self._emit(bits, self._history, advance=True)

    def _fresh_history(self) -> np.ndarray:
        return np.zeros(self._delay, dtype=np.float64)

    def _emit(self, bits: np.ndarray, history: np.ndarray, advance: bool) -> np.ndarray:
        """比特 → 成形频偏 → 相位积分 → IQ。

        history 为上一块末尾 delay 个 NRZ 样点；advance=True 时把本轮末尾的
        NRZ 与累积相位写回实例，使下一块能从同一个相位继续，而不是把相位归零。
        """
        if len(bits) == 0:
            return np.zeros(0, dtype=np.complex64)
        sps = self.mod.SAMPLES_PER_SYMBOL
        # NRZ 矩形脉冲：每个符号占 sps 个采样
        nrz = np.repeat(bits.astype(np.float64) * 2.0 - 1.0, sps)
        # 前缀 history 让块首的滤波窗口能看到上一块的末尾符号，后缀补零同理收尾。
        # 卷积本身有 delay 个采样的群延迟，加上前缀的 delay，有效区从 2*delay 起。
        padded = np.concatenate((history, nrz, np.zeros(self._delay)))
        start = 2 * self._delay
        shaped = np.convolve(padded, self._taps, mode="full")[start : start + len(nrz)]
        base = self._phase if advance else 0.0
        phase = base + np.cumsum(shaped * self._phase_inc)
        if advance:
            self._history = self._tail(nrz)
            # 相位按 2π 取模：累积到 1e9 弧度后 float64 的样点内精度会退化
            self._phase = float(phase[-1]) % (2 * math.pi)
        return (self.amplitude * np.exp(1j * phase)).astype(np.complex64)

    def _tail(self, nrz: np.ndarray) -> np.ndarray:
        """取块末 delay 个 NRZ 样点作为下一块的滤波历史；块比 delay 还短时左侧补零。"""
        if self._delay == 0:
            return self._history
        if len(nrz) >= self._delay:
            return nrz[-self._delay :].copy()
        return np.concatenate((np.zeros(self._delay - len(nrz)), nrz))


class GfskDemodulator:
    """复数基带 IQ → 判定比特（回环自检用）。

    仿真链路无噪声，采用"全相位搜索 + 访问码命中数选优"，
    不做定时恢复闭环，保证解码确定性。
    """

    def __init__(self, mod: ModulationParams) -> None:
        self.mod = mod

    def demodulate_to_bits(self, iq: np.ndarray, access_code_bits: List[int]) -> np.ndarray:
        sps = self.mod.SAMPLES_PER_SYMBOL
        # 相差鉴频得瞬时频率（归一化为符号极性量）
        # 差分会丢失 1 个采样，边缘补齐避免末尾符号被截断（整周期 9 包必须完整解出）
        f_inst = np.angle(iq[1:] * np.conj(iq[:-1]))
        f_inst = np.pad(f_inst, (0, sps), mode="edge")
        code_len = len(access_code_bits)
        code_arr = np.asarray(access_code_bits, dtype=np.uint8)
        frame_bits = 27 * 8
        best_bits: np.ndarray | None = None
        best_detects = -1
        for offset in range(sps):
            n_symbols = (len(f_inst) - offset) // sps
            windowed = f_inst[offset : offset + n_symbols * sps].reshape(n_symbols, sps)
            bits = (windowed.mean(axis=1) > 0).astype(np.uint8)
            # 相位判优：统计空口帧网格（每 216 bit）上的访问码检出数。
            # 不能用逐位匹配总数做度量——随机基线会淹没真实相关峰。
            detects = 0
            for start in range(0, n_symbols - code_len, frame_bits):
                dist = int(np.sum(bits[start : start + code_len] != code_arr))
                if dist <= 2:
                    detects += 1
            if detects > best_detects:
                best_detects = detects
                best_bits = bits
        assert best_bits is not None
        return best_bits
