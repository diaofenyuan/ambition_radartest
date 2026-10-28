# -*- coding: utf-8 -*-
"""PlutoSDR 发射链路独立探针。

界面显示"发射中"但频谱上看不到东西时，用它绕开业务逻辑直接把已知波形推到
设备，把问题范围切开：

- 探针有信号、程序没信号  → 程序里的设备绑定 / 模式 / 衰减
- 探针也没信号            → 端口、天线、功率、供电

用法：
    python tools/pluto_probe.py list
    python tools/pluto_probe.py tone --uri usb:1.6.5
    python tools/pluto_probe.py gfsk --uri usb:1.6.5 --wave info --seconds 10

tone 模式在中心频率 +100 kHz 处发单音而不是发 0 Hz：LO 泄漏在频谱仪上是一条
与设置无关的固定竖线，偏 100 kHz 才能确认"送到混频器的确实是我们的基带"。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 增益必须走这个统一入口：pyadi 的 TX 增益属性名随版本变，写错名字不报错、
# 只会在对象上挂一个没人读的属性，硬件增益原地不动。
from radar_tx.device.pluto_tx import (  # noqa: E402
    _set_tx_gain, _tx_gain_attr, to_dac_scale, tx_rf_bandwidth,
)

BLOCK = 100_000   # 与调度层一致的定长块：非循环 TX 缓冲建立后长度不可变


def _open(uri: str, tx_channel: str):
    """打开设备并关掉循环缓冲，失败时抛出带操作指引的异常。"""
    from radar_tx.device.libiio_runtime import ensure_libiio_available
    ok, detail = ensure_libiio_available()
    if not ok:
        raise SystemExit(f"[FAIL] {detail}")
    if not uri:
        from radar_tx.device.discovery import scan_pluto_devices
        found = scan_pluto_devices()
        if not found:
            raise SystemExit("[FAIL] 没扫描到 PlutoSDR，先用 list 子命令排查")
        uri = found[0].uri
        print(f"  自动选择设备 {uri}")
    try:
        import adi
    except Exception as exc:
        raise SystemExit(f"[FAIL] 导入 pyadi-iio 失败：{exc}")

    try:
        sdr = adi.Pluto(uri)
    except Exception as exc:
        raise SystemExit(f"[FAIL] 连接 {uri} 失败：{exc}\n       用 list 子命令确认 URI；"
                         f"USB 直连形如 usb:2.12.5，网口形如 ip:192.168.2.1")
    try:
        sdr.tx_destroy_buffer()
    except Exception:
        pass
    sdr.tx_cyclic_buffer = False
    if tx_channel:
        names = list(getattr(sdr, "tx_channel_names", []) or [])
        if tx_channel not in names:
            raise SystemExit(f"[FAIL] 没有 TX 通道 {tx_channel}，可选 {names}")
        sdr.tx_enabled_channels = [names.index(tx_channel)]
    return sdr


def _report(sdr, note: str) -> None:
    """回读设备实际生效的参数。

    设置成功不等于生效：LO 会被量化到可用频点，增益会被步进截断，通道号写错
    则数据进了另一个 SMA。回读是唯一能确认的方式。
    """
    gain_attr = _tx_gain_attr(sdr)
    channels = list(getattr(sdr, "tx_enabled_channels", []) or [])
    names = list(getattr(sdr, "tx_channel_names", []) or [])
    print(f"  {note}")
    print(f"    通道      {[names[i] for i in channels] if names else channels}")
    print(f"    采样率    {sdr.sample_rate / 1e6:.6f} MS/s")
    print(f"    本振      {sdr.tx_lo / 1e6:.6f} MHz")
    print(f"    RF 带宽   {sdr.tx_rf_bandwidth / 1e3:.1f} kHz")
    print(f"    硬件增益  {getattr(sdr, gain_attr, '?')} dB   （属性 {gain_attr}）")


def cmd_list(_args) -> None:
    from radar_tx.device.libiio_runtime import ensure_libiio_available
    ok, detail = ensure_libiio_available()
    if not ok:
        print(f"[FAIL] {detail}")
        return
    from radar_tx.device.discovery import last_scan_note, scan_pluto_devices
    devices = scan_pluto_devices()
    if not devices:
        print(last_scan_note() or "未发现 PlutoSDR。检查 USB 线与供电——供电不足时插上几秒就会掉线。")
        return
    for d in devices:
        print(f"{d.uri:24s} {d.model:12s} serial={d.serial}")


def _run(sdr, block_source, seconds: float, destroy) -> None:
    """按定长块持续推流，并打印每块耗时——耗时接近 100 ms 就是供不上。"""
    end = time.monotonic() + seconds
    count = 0
    try:
        while time.monotonic() < end:
            t0 = time.monotonic()
            sdr.tx(to_dac_scale(block_source()))
            count += 1
            print(f"\r  已推 {count} 块，单块耗时 {(time.monotonic() - t0) * 1e3:.1f} ms", end="")
    except KeyboardInterrupt:
        print("\n中断")
    except Exception as exc:
        print(f"\n[FAIL] 推流中断：{exc}")
    finally:
        destroy()
        print()


def cmd_tone(args) -> None:
    sdr = _open(args.uri, args.tx_channel)
    sdr.sample_rate = int(args.rate)
    sdr.tx_lo = int(args.freq)
    sdr.tx_rf_bandwidth = int(args.bw)
    _set_tx_gain(sdr, abs(args.gain))
    _report(sdr, "已下发")
    print(f"  → 频谱仪应在 {args.freq / 1e6 + 0.1:.3f} MHz 看到单音"
          f"（中心 {args.freq / 1e6:.3f} MHz 偏 +100 kHz）")

    n = np.arange(BLOCK)
    iq = (0.5 * np.exp(1j * 2 * np.pi * 1e5 * n / args.rate)).astype(np.complex64)
    _run(sdr, lambda: iq, args.seconds, lambda: sdr.tx_destroy_buffer())


def cmd_gfsk(args) -> None:
    from radar_tx.config import load_season_config
    from radar_tx.protocol.air_frame import AirStreamChunker
    from radar_tx.protocol.serial_frame import SerialFrameBuilder
    from radar_tx.waveform.gfsk import GfskModulator

    season = load_season_config()
    profile = season.profile(args.team, args.wave)
    builder = SerialFrameBuilder()
    chunker = AirStreamChunker(
        season.access_code_info if args.wave == "info" else season.access_code_interfere)
    modulator = GfskModulator(season.modulation, profile)

    sdr = _open(args.uri, args.tx_channel)
    sdr.sample_rate = int(season.modulation.SAMPLE_RATE_HZ)
    sdr.tx_lo = int(profile.freq_hz)
    # 模拟滤波带宽贴着信号带宽设会被严重削顶：jammer_3 的 250 kHz 实测比
    # 540 kHz 低 21 dB，波形根本出不去。默认走驱动层的统一策略，--rf-bw 可覆盖。
    sdr.tx_rf_bandwidth = (int(args.rf_bw) if args.rf_bw else
                           tx_rf_bandwidth(profile.bandwidth_hz,
                                           season.modulation.SAMPLE_RATE_HZ))
    _set_tx_gain(sdr, abs(args.gain))
    _report(sdr, f"已下发（{args.team}/{args.wave}，Δf={profile.freq_deviation_hz:.0f} Hz）")
    print(f"  → 频谱仪应看到约 {profile.bandwidth_hz / 1e3:.0f} kHz 宽的 2-GFSK 谱")

    cycle_samples = season.modulation.SAMPLE_RATE_HZ // season.cycles_per_sec
    state = {"pending": np.zeros(0, dtype=np.complex64)}

    def next_block() -> np.ndarray:
        if args.wave == "info":
            stream = (builder.pack_position([500] * 12)
                      + builder.pack_hp([300, 200, 150, 150, 0, 400])
                      + builder.pack_buff(bytes(41)))
        else:
            # 干扰波规则速率 1350 B/s = 135 B/周期。密钥帧只有 15 B，只喂它会让
            # 切片器每 2~3 个周期才凑出 1 个空口包，占空比掉到真实值的 1/9，
            # 接收端 200 ms 窗口里几乎全是静默——现象和"射频没通"一模一样。
            # 按规则用随机填充补齐到 135 B，才是真实波形密度。
            key = builder.pack_key("AB12CD")
            fill = np.random.default_rng(2026).integers(
                0, 256, season.jammer_bytes_per_cycle - len(key), dtype=np.uint8)
            stream = key + fill.tobytes()
        packets = chunker.feed(stream)
        iq = (modulator.modulate_stream(b"".join(packets))
              if packets else np.zeros(0, np.complex64))
        if len(iq) < cycle_samples:
            iq = np.pad(iq, (0, cycle_samples - len(iq)))
        data = np.concatenate((state["pending"], iq)) if state["pending"].size else iq
        take = min(len(data), BLOCK)
        block = np.zeros(BLOCK, dtype=np.complex64)
        block[:take] = data[:take]
        state["pending"] = data[take:].copy()
        return block

    _run(sdr, next_block, args.seconds, lambda: sdr.tx_destroy_buffer())


def main() -> None:
    p = argparse.ArgumentParser(description="PlutoSDR 发射链路探针")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp):
        sp.add_argument("--uri", default="", help="设备 URI，留空则自动用扫描到的第一台")
        sp.add_argument("--tx-channel", default="", help="留空=voltage0(TX1)；接 TX2 填 voltage1")
        sp.add_argument("--gain", type=float, default=10, help="衰减 dB，默认 10；越小越强")
        sp.add_argument("--seconds", type=float, default=20, help="发射时长，默认 20 秒")

    lst = sub.add_parser("list", help="列出可见的 PlutoSDR")
    lst.set_defaults(func=cmd_list)

    tone = sub.add_parser("tone", help="+100kHz 单音，先验端口与功率")
    add_common(tone)
    tone.add_argument("--freq", type=float, default=433.2e6, help="本振频率 Hz")
    tone.add_argument("--bw", type=float, default=540e3, help="RF 带宽 Hz")
    tone.add_argument("--rate", type=float, default=1e6, help="采样率")
    tone.set_defaults(func=cmd_tone)

    gfsk = sub.add_parser("gfsk", help="按赛季参数发真实 2-GFSK 调制波")
    add_common(gfsk)
    gfsk.add_argument("--team", default="red", choices=["red", "blue"])
    gfsk.add_argument("--wave", default="info",
                      choices=["info", "jammer_1", "jammer_2", "jammer_3"])
    gfsk.add_argument("--rf-bw", type=float, default=0,
                      help="覆盖 TX 模拟滤波带宽 Hz；留空=信号带宽。"
                           "窄带波形（如 jammer_3 的 250k）留空会被削掉约 21 dB")
    gfsk.set_defaults(func=cmd_gfsk)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
