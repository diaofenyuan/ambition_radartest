# -*- coding: utf-8 -*-
"""发射节拍与待发缓冲的回归测试。

信息波按规则是 1400 B/s，15 字节一包——不是 15 的整数倍，因此每 3 个周期
会出现一次 10 包（101520 采样）的超长周期，其余周期是 9 包（91368 采样）。
一旦把 9 包周期补零抬到整周期 100000，平均产出就变成 100507 > 设备消耗的
100000，待发缓冲会无限增长：实测 300 周期积压 152 ms，一小时滞后 18 秒。
这里把"长跑不积压"固化成断言。
"""

import numpy as np

from radar_tx.config import load_devices_config, load_season_config
from radar_tx.protocol.air_frame import AirStreamChunker
from radar_tx.protocol.serial_frame import SerialFrameBuilder
from radar_tx.scheduler.tx_service import TxService
from radar_tx.waveform.gfsk import GfskModulator
from radar_tx.waveform.stream import modulate_air_packets


def _simulate_backlog(wave: str, cycles: int = 300) -> int:
    """按发射线程的实际节奏模拟若干周期，返回最终待发采样数。"""
    season = load_season_config()
    devices = load_devices_config()
    # 不传 scanner 会去扫真实设备，测试不该依赖硬件在场
    svc = TxService(season, devices, device_scanner=lambda: [])
    block = season.modulation.SAMPLE_RATE_HZ // season.cycles_per_sec
    builder = SerialFrameBuilder()
    chunker = AirStreamChunker(
        season.access_code_info if wave == "info" else season.access_code_interfere)
    mod = GfskModulator(
        season.modulation,
        season.profile("red", "info" if wave == "info" else "jammer_3"))
    pending = 0
    for i in range(cycles):
        stream = (svc._build_info_stream(builder) if wave == "info"
                  else svc._build_jammer_stream(builder, i))
        iq = modulate_air_packets(chunker.feed(stream), mod)
        data = pending + len(iq)
        pending = data - min(data, block)
    return pending


def test_info_waveform_has_ten_packet_cycles():
    """确认前提成立：信息波确实会出现超过整周期的超长周期。"""
    season = load_season_config()
    block = season.modulation.SAMPLE_RATE_HZ // season.cycles_per_sec
    builder = SerialFrameBuilder()
    chunker = AirStreamChunker(season.access_code_info)
    mod = GfskModulator(season.modulation, season.profile("red", "info"))
    svc = TxService(season, load_devices_config(), device_scanner=lambda: [])
    lengths = []
    for _ in range(6):
        iq = modulate_air_packets(chunker.feed(svc._build_info_stream(builder)), mod)
        lengths.append(len(iq))
    assert max(lengths) > block, "应当出现超过整周期的 10 包周期"
    assert min(lengths) < block, "应当出现不足整周期的 9 包周期"


def test_info_backlog_stays_bounded():
    # 一个块以内（≤100000）都正常；这里收紧到 2 万采样（20 ms）盯住趋势
    assert _simulate_backlog("info") <= 20_000


def test_jammer_backlog_stays_bounded():
    assert _simulate_backlog("jammer") <= 20_000
