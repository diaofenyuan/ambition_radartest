# -*- coding: utf-8 -*-
"""连续发射波形测试：包边界与周期边界不得出现相位跳变。

规则附录 1 明确"无固定包间隔，数据到达即发射"，空口包之间没有保护间隔，
整条数据流应当是一段连续波形。早先实现逐包调制再首尾相接，每个包边界都把
相位归零，实测瞬时频偏会跳出设计值（红方广播源 -446 kHz，是 Δf 的 1.8 倍），
且每包首尾各有一个符号的频偏被削掉一半。
"""

import numpy as np
import pytest

from radar_tx.config import load_season_config
from radar_tx.protocol.air_frame import AIR_FRAME_LEN, ACCESS_CODE_INFO, AirStreamChunker
from radar_tx.waveform.gfsk import GfskModulator, gaussian_taps
from radar_tx.waveform.stream import modulate_air_packets

SEASON = load_season_config()
SPB = AIR_FRAME_LEN * 8 * SEASON.modulation.SAMPLES_PER_SYMBOL  # 单个空口包的样点数
DELAY = (len(gaussian_taps(SEASON.modulation)) - 1) // 2        # 高斯滤波器群延迟


def _packets(n_bytes: int = 150):
    """一个周期的空口包（150B → 10 个包，包内容各不相同）。"""
    return AirStreamChunker(ACCESS_CODE_INFO).feed(bytes(range(n_bytes)))


def _freq(iq: np.ndarray) -> np.ndarray:
    """瞬时频偏（Hz）：相邻样点相位差。"""
    return np.angle(iq[1:] * np.conj(iq[:-1])) * SEASON.modulation.SAMPLE_RATE_HZ / (2 * np.pi)


@pytest.mark.parametrize("team,wave", [
    ("red", "info"), ("red", "jammer_1"), ("red", "jammer_2"), ("red", "jammer_3"),
    ("blue", "info"), ("blue", "jammer_1"), ("blue", "jammer_2"), ("blue", "jammer_3"),
])
def test_instantaneous_deviation_never_exceeds_design(team, wave):
    """瞬时频偏必须恒在 ±Δf 内，越界即意味包边界存在相位跳变。

    高斯滤波是对 ±1 的凸组合（抽头非负、和为 1），成形后的频偏不可能超出设计
    频偏；只要跨包相位连续，这个界就自然成立。逐包调制会在这里被抓住。
    """
    profile = SEASON.profile(team, wave)
    iq = modulate_air_packets(_packets(), GfskModulator(SEASON.modulation, profile))
    peak = float(np.abs(_freq(iq)).max())
    assert peak <= profile.freq_deviation_hz * 1.0001, f"{team}/{wave} 瞬时频偏越界 {peak:.0f} Hz"


def test_cross_cycle_keeps_filter_state_and_phase():
    """同一调制器连发两个周期，等效于一次发两周期。

    除第一块末尾 delay 个样点（整段收尾需要补零，属设计内差异）外，
    两周期波形必须逐点一致——这证明滤波器尾样本与累积相位确实跨周期传递了。
    """
    packets = _packets()
    profile = SEASON.profile("red", "info")
    whole = modulate_air_packets(packets * 2, GfskModulator(SEASON.modulation, profile))

    modulator = GfskModulator(SEASON.modulation, profile)
    split = np.concatenate([modulate_air_packets(packets, modulator) for _ in range(2)])

    cycle_samples = len(packets) * SPB
    assert len(whole) == len(split) == 2 * cycle_samples

    diff = np.abs(_freq(whole) - _freq(split))
    # 累积相位不同只造成整体相位旋转，鉴频结果与绝对相位无关，差异只剩 float32 噪声
    # （实测 ~0.04 Hz，相对 248.7 kHz 频偏约 1.6e-7）。真正的相位跳变是几十 kHz 量级。
    # 分块末尾有 delay 个样点的滤波窗口越过了块尾（整段收尾时补零），
    # 从 N-delay-1 起才允许出现差异。
    assert diff[: cycle_samples - DELAY - 1].max() < 1.0
    assert diff[cycle_samples:].max() < 1.0


def test_reset_makes_next_block_start_cold():
    """reset 之后必须与全新调制器输出一致：发射中断期间射频是静的，
    接着上一轮的相位发会在开头凭空多出一个相位跳变。"""
    packets = _packets()
    profile = SEASON.profile("red", "info")
    data = b"".join(packets)

    used = GfskModulator(SEASON.modulation, profile)
    used.modulate_stream(data)
    used.reset()
    after_reset = used.modulate_stream(data)

    fresh = GfskModulator(SEASON.modulation, profile).modulate_stream(data)
    assert np.array_equal(after_reset, fresh)
