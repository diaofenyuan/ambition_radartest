# -*- coding: utf-8 -*-
"""发射数据缩放的回归测试。

pyadi 的 tx() 对复数样点只做 astype(int16)，不做归一化。波形层输出的是
±1 归一化 IQ，驱动层若不换算，落到 DAC 上就是全零——真机上表现为
"界面显示发射中、频谱仪上什么都没有"，而仿真模式完全正常，无硬件时
根本发现不了。这个测试把换算关系固化下来。
"""

import numpy as np

from radar_tx.device.pluto_tx import (
    DAC_FULL_SCALE, TX_RF_BANDWIDTH_MIN_HZ, to_dac_scale, tx_rf_bandwidth,
)


def test_scale_reaches_dac_range():
    iq = np.array([0.5 + 0.5j, -1.0 + 0j, 0j], dtype=np.complex64)
    out = to_dac_scale(iq)
    assert out.dtype == np.complex64
    assert np.isclose(out[0].real, 0.5 * DAC_FULL_SCALE)
    assert np.isclose(out[1].real, -DAC_FULL_SCALE)
    assert out[2] == 0


def test_scale_survives_int16_cast():
    """换算后 astype(int16) 不能退化成全零——那才是 DAC 真正看到的数。

    pyadi 是实部、虚部分别 astype 的，这里照做，避免把复数直接转整型。
    """
    iq = (0.5 * np.exp(1j * np.linspace(0, 2 * np.pi, 64))).astype(np.complex64)
    out = to_dac_scale(iq)
    i = out.real.astype(np.int16)
    q = out.imag.astype(np.int16)
    assert np.count_nonzero(i) > 0 and np.count_nonzero(q) > 0
    assert max(np.abs(i.astype(np.int32)).max(), np.abs(q.astype(np.int32)).max()) > 1000


def test_normalised_input_would_be_silent():
    """把 bug 本身写成断言：不换算直接用 ±0.5 的实部转 int16 就是全零。"""
    iq = (0.5 * np.exp(1j * np.linspace(0, 2 * np.pi, 64))).astype(np.complex64)
    assert np.count_nonzero(iq.real.astype(np.int16)) == 0


def test_no_overflow_at_full_scale():
    iq = np.ones(8, dtype=np.complex64)
    out = to_dac_scale(iq)
    assert out.real.astype(np.int16).max() <= 32767


def test_rf_bandwidth_keeps_headroom_for_narrow_waveforms():
    """窄带波形要留模拟滤波余量：250k 直接照抄会被削掉约 21 dB。"""
    assert tx_rf_bandwidth(250_000, 1_000_000) == TX_RF_BANDWIDTH_MIN_HZ
    # 宽带 waveform 原样保留，不多放带外噪声
    assert tx_rf_bandwidth(940_000, 1_000_000) == 940_000
    # 不超过采样率
    assert tx_rf_bandwidth(2_000_000, 1_000_000) == 1_000_000
