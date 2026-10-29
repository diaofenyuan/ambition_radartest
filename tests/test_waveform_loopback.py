# -*- coding: utf-8 -*-
"""波形回环测试：业务数据 → 帧 → GFSK → 回环解调 → 业务数据 全链路一致性。

这是模拟发射系统最关键的正确性链路：波形层任何参数错配（频偏、符号率、
比特序、滤波群延迟）都会在这里暴露。
调制参数与规则手册 V2.2.0 表 5-23 / 附录 1 对齐校验亦在此覆盖。
"""

import math
import os

import pytest

from radar_tx.config import ModulationParams, RfProfile, load_season_config
from radar_tx.protocol.air_frame import ACCESS_CODE_INTERFERE, AirStreamChunker
from radar_tx.protocol.serial_frame import (
    CMD_FIELD_STATUS, CMD_INTERFERENCE_KEY, CMD_ROBOT_AMMO, CMD_ROBOT_BUFF,
    CMD_ROBOT_HP, CMD_ROBOT_POSITION, SerialFrameBuilder,
)
from radar_tx.sim.loopback import LoopbackDecoder
from radar_tx.waveform.gfsk import GfskModulator
from radar_tx.waveform.stream import modulate_air_packets

SEASON = load_season_config()


def test_rf_profiles_match_rule_table_5_23():
    """与官方规则手册表 5-23 逐项核对：频点 / 带宽 / 功率 / 调频灵敏度。"""
    expected = {
        ("red", "info"): (433_200_000, 540_000, -60, 1.5628),
        ("red", "jammer_1"): (432_200_000, 940_000, -10, 2.8194),
        ("red", "jammer_2"): (432_500_000, 860_000, -10, 2.5681),
        ("red", "jammer_3"): (432_800_000, 250_000, -10, 0.6517),
        ("blue", "info"): (433_920_000, 540_000, -60, 1.5628),
        ("blue", "jammer_1"): (434_920_000, 940_000, -10, 2.8194),
        ("blue", "jammer_2"): (434_620_000, 860_000, -10, 2.5681),
        ("blue", "jammer_3"): (434_320_000, 250_000, -10, 0.6517),
    }
    for (team, wave), (freq, bw, power, sens) in expected.items():
        p = SEASON.profile(team, wave)
        assert (p.freq_hz, p.bandwidth_hz, p.power_dbm) == (freq, bw, power), (team, wave)
        got = 2 * math.pi * p.freq_deviation_hz / SEASON.modulation.SAMPLE_RATE_HZ
        assert round(got, 4) == sens, (team, wave, got)


def test_modulation_params_match_rule():
    mod = SEASON.modulation
    assert mod.SAMPLE_RATE_HZ == 1_000_000
    assert mod.SAMPLES_PER_SYMBOL == 47
    assert mod.GFSK_BT == 0.35
    assert mod.GAUSSIAN_SPAN_SYMBOLS == 4
    # 附录 1：红方广播源 bit=1 瞬时频偏 +248,723Hz
    dev = SEASON.profile("red", "info").freq_deviation_hz
    assert round(dev) == 248_723


def test_freq_deviation_follows_loaded_symbol_rate():
    """频偏必须由加载后的码率推导，而不是 ModulationParams 的类默认值。

    早先 freq_deviation_hz 直接读类属性，等价于永远按 1MS/s ÷ 47 计算：
    换赛季把 YAML 的 SPS 或采样率改掉后，频偏会静默偏离规则公式，
    而 config.py 的设计目标恰恰是"换赛季只换 YAML"。
    """
    mod = ModulationParams(SAMPLE_RATE_HZ=1_000_000, SAMPLES_PER_SYMBOL=50)
    profile = RfProfile(freq_hz=433_200_000, bandwidth_hz=540_000, power_dbm=-60,
                        bit_rate_hz=mod.bit_rate_hz)
    # Δf = 270000 - 20000
    assert round(profile.freq_deviation_hz) == 250_000


@pytest.mark.parametrize("team", ["red", "blue"])
def test_info_wave_loopback(team):
    # 信息波周期流：5 命令帧 = 33+21+19+21+56 = 150B（各命令 10Hz）
    builder = SerialFrameBuilder()
    stream = (builder.pack_position(range(12))
              + builder.pack_hp([100, 0, 300, 400, 0, 600])
              + builder.pack_ammo([10, 20, 30, 0, 50])
              + builder.pack_field_status(100, 2000, 3, 0)
              + builder.pack_buff(os.urandom(47)))
    assert len(stream) == 150
    # 连续喂入三个周期：450B 恰好 30 个空口包，且跨周期帧能完整恢复
    chunker = AirStreamChunker(SEASON.access_code_info)
    packets = []
    for _ in range(3):
        packets.extend(chunker.feed(stream))
    assert len(packets) == 30
    profile = SEASON.profile(team, "info")
    iq = modulate_air_packets(packets, GfskModulator(SEASON.modulation, profile))

    result = LoopbackDecoder(SEASON).decode(iq, is_info=True)
    expected_cmds = [CMD_ROBOT_POSITION, CMD_ROBOT_HP, CMD_ROBOT_AMMO, CMD_FIELD_STATUS, CMD_ROBOT_BUFF]
    assert [f.cmd_id for f in result.frames] == expected_cmds * 3
    assert len(result.frames[4].payload) == 47
    assert result.frames[1].payload == bytes.fromhex(
        "640000002c01900100005802")  # 工程位与第 5 项保留位为 0


@pytest.mark.parametrize("team", ["red", "blue"])
@pytest.mark.parametrize("level", [1, 2, 3])
def test_jammer_wave_loopback_with_sof_offset(team, level):
    # 干扰波周期流：135B = 1 帧 0x0A06 + 随机填充（规则 1350B/s）
    builder = SerialFrameBuilder()
    key_frame = builder.pack_key("AB12CD")
    offset = 7  # 非对齐偏移，验证滑窗恢复
    stream = os.urandom(offset) + key_frame + os.urandom(135 - offset - len(key_frame))
    packets = AirStreamChunker(ACCESS_CODE_INTERFERE).feed(stream)
    assert len(packets) == 9
    profile = SEASON.profile(team, f"jammer_{level}")
    iq = modulate_air_packets(packets, GfskModulator(SEASON.modulation, profile))

    result = LoopbackDecoder(SEASON).decode(iq, is_info=False)
    key_frames = [f for f in result.frames if f.cmd_id == CMD_INTERFERENCE_KEY]
    assert len(key_frames) == 1
    assert key_frames[0].payload == b"AB12CD"
