# -*- coding: utf-8 -*-
"""收发链路体检：一块板循环发单音，另一块板收，用 FFT 量化信号强度。

为什么需要它：GFSK 解不出帧时，无法区分"射频没通""信号太弱""解调参数错"。
单音是功率最集中的探针信号，只要看它在接收端比噪声高多少 dB，就能直接算出
符号带宽上还剩多少余量，也能在换天线/换接头/换 USB 口之后用同一套参数对比。

判定门槛按 2-FSK 反推：符号率 21.3 kHz，解调需要符号 SNR 约 +7 dB。
单音 SNR（在 3.8 Hz 分辨率上）换算到符号带宽要减去 10·log10(21300/3.8) ≈ 37 dB，
再预留 10 dB 实现损耗，因此单音 SNR 低于 45 dB 时基本解不出帧。

用法：
    python tools/link_check.py --tx-uri usb:2.13.5 --rx-uri usb:2.14.5
    python tools/link_check.py --freq 433.2e6 --tx-gain 0 --rx-gain 71
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from radar_tx.config import load_season_config

TONE_HZ = 100_000      # 单音偏置：避开零中频的 LO 泄漏，否则两者在频谱上会重叠
FS = 1_000_000
BUFFER = 100_000


def _open(uri: str, note: str):
    from radar_tx.device.libiio_runtime import ensure_libiio_available
    ok, detail = ensure_libiio_available()
    if not ok:
        raise SystemExit(f"[FAIL] {detail}")
    if not uri:
        from radar_tx.device.discovery import scan_pluto_devices
        found = scan_pluto_devices()
        if not found:
            raise SystemExit("[FAIL] 没扫描到 PlutoSDR")
        uri = found[0].uri
        print(f"  {note} 自动选择 {uri}")
    import adi
    try:
        return adi.Pluto(uri)
    except Exception as exc:
        raise SystemExit(f"[FAIL] 连接 {note} {uri} 失败：{exc}")


def _start_tone(args) -> tuple:
    """打开发射板并持续发单音，返回 (设备, 提示串, 停止事件, 线程)。

    不用 tx_cyclic_buffer：循环缓冲提交后只看见 LO 泄漏、基带波形不出现在
    输出上（实测衰减从 60 dB 扫到 0 dB，接收电平纹丝不动）。改成后台线程
    反复 tx() 推同一块数据，行为才和 tools/pluto_probe.py 里验证过的一致。
    """
    import threading

    from radar_tx.device.pluto_tx import _set_tx_gain, _tx_gain_attr, to_dac_scale
    tx = _open(args.tx_uri, "发射板")
    try:
        tx.tx_destroy_buffer()
    except Exception:
        pass
    tx.tx_cyclic_buffer = False
    tx.sample_rate = FS
    tx.tx_lo = int(args.freq)
    tx.tx_rf_bandwidth = int(args.bw)
    _set_tx_gain(tx, abs(args.tx_gain))
    n = np.arange(BUFFER)
    tone = to_dac_scale((0.5 * np.exp(1j * 2 * np.pi * TONE_HZ * n / FS)).astype(np.complex64))
    # 设置成功不等于生效：LO 会被量化、增益会被步进截断，必须回读确认
    info = (f"TX {tx.tx_lo / 1e6:.6f} MHz  衰减 {getattr(tx, _tx_gain_attr(tx), '?')} dB"
            f"  RF 带宽 {tx.tx_rf_bandwidth / 1e3:.0f} kHz  持续发射中")

    stop = threading.Event()

    def _push() -> None:
        while not stop.is_set():
            try:
                tx.tx(tone)
            except Exception:
                break

    thread = threading.Thread(target=_push, daemon=True)
    thread.start()
    return tx, info, stop, thread


def _measure(args) -> None:
    from radar_tx.device.pluto_tx import _tx_gain_attr  # noqa: F401

    tx, tx_info, stop, thread = _start_tone(args)
    rx = _open(args.rx_uri, "接收板")
    rx.sample_rate = FS
    # 本振直接对准信号：单音在 +100 kHz，离 DC 泄漏足够远，不需要再错开本振
    rx.rx_lo = int(args.freq)
    rx.rx_rf_bandwidth = int(args.bw)
    rx.gain_control_mode_chan0 = "manual"
    rx.rx_hardwaregain_chan0 = float(args.rx_gain)
    rx.rx_buffer_size = 262144
    print(f"  {tx_info}")
    print(f"  RX {rx.rx_lo / 1e6:.6f} MHz  增益 {rx.rx_hardwaregain_chan0:.0f} dB"
          f"  RF 带宽 {rx.rx_rf_bandwidth / 1e3:.0f} kHz")
    print(f"  频率 {args.freq / 1e6:.3f} MHz  单音应在 +{TONE_HZ / 1e3:.0f} kHz")

    acc = None
    rms_last = 0.0
    for _ in range(args.segments):
        x = np.array(rx.rx(), copy=True).astype(np.complex128)
        # pyadi 可能给 ±1 归一化浮点，也可能给原始 int16 计数，按"是否为整数"判别
        scale = 32768.0 if np.abs(x - np.round(x)).max() < 1e-9 else 1.0
        x = x / scale
        x = x - np.mean(x)
        rms_last = float(np.sqrt(np.mean(np.abs(x) ** 2)))
        power = np.abs(np.fft.fftshift(np.fft.fft(x * np.hanning(len(x))))) ** 2
        acc = power if acc is None else acc + power

    stop.set()
    thread.join(timeout=1.5)
    try:
        tx.tx_destroy_buffer()
    except Exception:
        pass

    level = 10 * np.log10(acc / args.segments + 1e-30)
    freqs = np.fft.fftshift(np.fft.fftfreq(262144, 1 / FS))
    noise = float(np.median(level))
    # 两板晶振差在 433 MHz 上可达 ±10 kHz，搜索窗要放宽到 ±15 kHz
    band = np.abs(freqs - TONE_HZ) <= 15e3
    peak = float(level[band].max())
    peak_hz = float(freqs[band][int(np.argmax(level[band]))])
    snr = peak - noise

    print()
    print(f"  去直流后 RMS {rms_last:.6f}（满量程 1.0）")
    print(f"  噪声中位 {noise:.1f} dB")
    print(f"  单音峰   {peak:.1f} dB @ {peak_hz / 1e3:+.2f} kHz（相对 {TONE_HZ / 1e3:.0f} kHz"
          f" 偏 {(peak_hz - TONE_HZ) / 1e3:+.2f} kHz）")
    print(f"  单音 SNR {snr:+.1f} dB")

    print()
    if snr >= 45:
        print("[OK] 链路余量充足，这个强度足够解出 2-GFSK 帧。")
    elif snr >= 30:
        print("[WARN] 链路偏弱：单音能收到，但换算到符号带宽余量不足，"
              "2-GFSK 大概率解不出。优先缩短距离或换谐振天线。")
    elif snr >= 12:
        print("[WARN] 只收到微弱载波：射频通路存在但衰减过大，"
              "检查天线是否拧到底、是否谐振在目标频段。")
    else:
        print("[FAIL] 几乎没有信号：检查发射板是否真在发、天线是否接在 "
              "TX1A / RX1A、本振是否对准。")


def main() -> None:
    season = load_season_config()
    default_freq = season.profile("red", "jammer_3").freq_hz
    p = argparse.ArgumentParser(description="PlutoSDR 收发链路体检（单音 SNR）")
    p.add_argument("--tx-uri", default="", help="发射板 URI")
    p.add_argument("--rx-uri", default="", help="接收板 URI")
    p.add_argument("--freq", type=float, default=float(default_freq),
                   help=f"收发频点 Hz，默认 {default_freq / 1e6:.1f} MHz")
    p.add_argument("--bw", type=float, default=540e3, help="RF 带宽 Hz，默认 540k")
    p.add_argument("--tx-gain", type=float, default=0, help="发射衰减 dB，默认 0（最大功率）")
    p.add_argument("--rx-gain", type=float, default=60, help="接收增益 dB，默认 60")
    p.add_argument("--segments", type=int, default=4, help="平均段数，默认 4")
    args = p.parse_args()
    _measure(args)


if __name__ == "__main__":
    main()
