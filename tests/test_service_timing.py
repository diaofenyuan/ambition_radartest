# -*- coding: utf-8 -*-
"""发射节拍与待发缓冲的回归测试。

信息波 2027 协议是 150B/周期，恰好 10 个空口包（101520 采样），
在 1MS/s / SPS=47 下已经超过 100ms 周期预算（100000 采样）——
这是已知且被显式告警的差距，射频参数要等 2027 规则手册发布后才能消除。
干扰波 135B/周期仍是 9 包（91368 采样），不能因信息波变长而积压。
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


def test_info_waveform_is_ten_packets_every_cycle():
    """150B/周期 = 10 个空口包 = 101520 采样，每个周期都超出 100ms 预算。"""
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
    assert all(length == 101_520 for length in lengths)
    assert lengths[0] > block


def test_info_capacity_overrun_is_exposed():
    """业务帧变长后 1MS/s/SPS=47 装不下 150B/10Hz，服务必须显式告警。"""
    svc = TxService(load_season_config(), load_devices_config(), device_scanner=lambda: [])
    try:
        cap = svc.snapshot()["rules"]["capacity"]
        assert cap["ok"] is False
        assert cap["payload_bytes"] == 150
        assert cap["required_bps"] > cap["bit_rate_bps"]
        assert cap["air_ms"] > cap["budget_ms"]
        assert any("空口容量告警" in line for line in svc.snapshot()["logs"])
    finally:
        svc.shutdown()


def test_jammer_backlog_stays_bounded():
    assert _simulate_backlog("jammer") <= 20_000
