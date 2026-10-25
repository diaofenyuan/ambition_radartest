# -*- coding: utf-8 -*-
"""PlutoSDR 接收探针：收一收，看发射端到底能不能被解出来。

和 tools/pluto_probe.py 配对：发射板跑 gfsk 模式，接收板跑本脚本。

接收侧做三件真实接收机必须做的事，少一件都可能解不出：

1. 本振偏置 + 数字搬移。RX 的 LO 不能和 TX 同频——Pluto 是零中频，同频时信号
   正好压在 DC 上，被 LO 泄漏和白噪声淹没。默认把 LO 抬高 300 kHz，让信号落在
   基带 -300 kHz，再乘一个 NCO 把它搬回 0。只搬 LO 不搬基带是不行的：
   2-FSK 的判决门限是 0，偏 300 kHz 等于把所有符号判成同一个值。
2. 去直流。残余 LO 泄漏是加性复偏置，必须在混频前减掉，否则跟着一起被搬走。
3. 判决门限固定为 0，不用自适应。数据 0/1 不平衡会让中位数偏离 0，反而判错；
   两块板晶振差几十 ppm 在 433 MHz 上只有几 kHz，相对 ±248 kHz 频偏可以忽略。

用法：
    python tools/pluto_rx_probe.py                      # 自动选板，收 5 秒
    python tools/pluto_rx_probe.py --uri usb:2.12.5 --gain 40 --seconds 8
    python tools/pluto_rx_probe.py --offset 0           # 本振直接对准信号
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from radar_tx.config import load_season_config
from radar_tx.protocol.air_frame import bytes_to_bits, extract_air_payloads
from radar_tx.protocol.serial_frame import SerialFrameDecoder


def _open(uri: str):
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
    import adi
    sdr = adi.Pluto(uri)
    print(f"接收板 {uri}")
    return sdr


def count_access_hits(bits: np.ndarray, code_bits: np.ndarray, max_bit_errors: int = 2) -> int:
    """统计比特流里 Access Code 的命中次数，逐位滑窗、不假设帧网格。

    发射端每周期补零、跨周期切片，帧边界会落在任意比特位置上。按 216 bit
    固定网格搜索时，只有"采集起点恰好对齐"的 1/216 情况能命中，其余时候
    明明收到信号也会得到 detects=0，整段样本被丢弃。
    """
    n, code_len = len(bits), len(code_bits)
    if n < code_len:
        return 0
    b = bits.astype(np.int8) * 2 - 1
    c = code_bits.astype(np.int8) * 2 - 1
    # 相关值 = 匹配位数 - 不匹配位数 = code_len - 2×错误位数
    corr = np.convolve(b, c[::-1], mode="valid")
    return int(np.count_nonzero(corr >= code_len - 2 * max_bit_errors))


def demodulate(iq: np.ndarray, sps: int, code_bits: np.ndarray):
    """鉴频 + 全相位搜索，返回（比特流, 最佳 offset, 频偏标记, Access Code 检出数）。

    iq 必须已经搬到基带中心（见 shift_to_baseband）。
    """
    f = np.angle(iq[1:] * np.conj(iq[:-1]))
    f = np.pad(f, (0, sps), mode="edge")
    best_bits, best_detects, best_off, best_cfo = None, -1, 0, 0.0
    for off in range(sps):
        n = (len(f) - off) // sps
        if n < 216:
            continue
        sym = f[off : off + n * sps].reshape(n, sps).mean(axis=1)
        bits = (sym > 0).astype(np.uint8)       # 门限固定 0：这是 2-FSK 的判决面
        detects = count_access_hits(bits, code_bits)
        if detects > best_detects:
            best_bits, best_detects, best_off = bits, detects, off
            best_cfo = float(np.median(sym))     # 仅用于诊断显示
    return best_bits, best_off, best_cfo, best_detects


def shift_to_baseband(iq: np.ndarray, offset_hz: float, fs: int) -> np.ndarray:
    """去直流后把信号从 -offset 搬回 0。

    LO 抬高 offset 时信号落在基带 -offset，DC 泄漏留在 0。先减均值去 DC，
    再乘 +offset 的 NCO——信号回到 0 供判决，残余的直流则被搬到 +offset 一侧。
    """
    x = iq - np.mean(iq)
    if not offset_hz:
        return x
    n = np.arange(len(x))
    return x * np.exp(1j * 2 * np.pi * offset_hz * n / fs)


def run(args) -> None:
    season = load_season_config()
    is_info = args.wave == "info"
    profile = season.profile(args.team, args.wave)
    code_bits = np.asarray(bytes_to_bits(
        season.access_code_info if is_info else season.access_code_interfere), dtype=np.uint8)
    sps = season.modulation.SAMPLES_PER_SYMBOL
    fs = season.modulation.SAMPLE_RATE_HZ

    sdr = _open(args.uri)
    sdr.sample_rate = int(fs)
    sdr.rx_lo = int(profile.freq_hz + args.offset)
    # RX 的 RF 带宽以本振为中心。把本振挪开 offset 之后，通带必须仍然覆盖
    # 「信号中心 ± 信号带宽/2」，否则 2-GFSK 的边带会被模拟滤波器直接切掉——
    # 本振偏 300 kHz 却只开 540 kHz 带宽时，信号中心正好落在通带边缘外。
    rx_bw = int(max(profile.bandwidth_hz,
                    2 * (abs(args.offset) + profile.bandwidth_hz / 2)))
    sdr.rx_rf_bandwidth = rx_bw
    sdr.gain_control_mode_chan0 = "manual"
    sdr.rx_hardwaregain_chan0 = float(args.gain)
    sdr.rx_buffer_size = 200_000
    print(f"  本振 {sdr.rx_lo / 1e6:.3f} MHz（相对信号 {args.offset / 1e3:+.0f} kHz）"
          f"  带宽 {sdr.rx_rf_bandwidth / 1e3:.0f} kHz  增益 {sdr.rx_hardwaregain_chan0:.0f} dB")
    rch = list(getattr(sdr, "rx_enabled_channels", []) or [])
    rnames = list(getattr(sdr, "rx_channel_names", []) or [])
    print(f"  RX 通道 {[rnames[i] for i in rch] if rnames else rch}"
          f"（voltage0=RX1A，接错口收不到）")
    print(f"  等信号中，{args.seconds:.0f} 秒内任何时刻发射板发波都能收到…")

    decoder = SerialFrameDecoder()
    total_payloads, frames, cycles = 0, [], 0
    end = time.monotonic() + args.seconds
    peak_rms = peak_sig = 0.0
    while time.monotonic() < end:
        iq = np.array(sdr.rx(), copy=True)
        # pyadi 的 rx() 在不同版本/固件下可能给归一化到 ±1 的浮点，也可能给原始
        # int16 计数（满量程 ±32768，噪声底只有 ~0.6）。按"值是否为整数"判断：
        # 用幅度阈值判断会失灵——低增益时峰值只有 2~5，照样是原始计数。
        scale = 32768.0 if np.abs(iq - np.round(iq)).max() < 1e-9 else 1.0
        iq = (iq / scale).astype(np.complex64)
        cycles += 1
        rms = float(np.sqrt(np.mean(np.abs(iq) ** 2)))
        peak_rms = max(peak_rms, rms)
        x = shift_to_baseband(iq, args.offset, fs)
        # 去直流后的幅度才是"信号有多强"：零中频接收机的原始 RMS 里，
        # 自身 LO 泄漏和直流失调往往占掉大半，容易误判成"收到了信号"。
        peak_sig = max(peak_sig, float(np.sqrt(np.mean(np.abs(x) ** 2))))
        bits, off, cfo, detects = demodulate(x.astype(np.complex64), sps, code_bits)
        if bits is None or detects == 0:
            continue
        payloads = extract_air_payloads(bits.tolist(), code_bits.tolist(), max_bit_errors=2)
        total_payloads += len(payloads)
        for p in payloads:
            frames.extend(decoder.feed(p))
        print(f"  [第{cycles}次] RMS {rms:.4f}  频偏 {cfo * fs / (2 * np.pi):+.1f} kHz  "
              f"Access Code {detects} 个  空口载荷 {len(payloads)} 个  业务帧 {len(frames)} 个")
        if frames and len(frames) >= args.stop_after:
            break

    print()
    if not frames:
        print(f"[FAIL] 没解出业务帧。原始 RMS {peak_rms:.5f}，去直流后 {peak_sig:.5f}")
        if peak_sig < 1e-3:
            print("   → 去直流后几乎没能量：信号没进来。检查发射板是否在发、"
                  "收发天线是否接在 TX1A / RX1A、本振是否对准")
        elif peak_sig > 0.5:
            print("   → 去直流后幅度仍接近满量程：接收机饱和。把 --gain 降到 0，"
                  "或把发射侧 --gain（衰减）加大到 70~89")
        else:
            print("   → 收到了像样的信号但解不出：试 --offset 0，或检查带宽是否覆盖"
                  "「信号中心 ± 270 kHz」")
        return
    print(f"[OK] 解出 {len(frames)} 个业务帧：")
    for f in frames[:12]:
        print(f"   0x{f.cmd_id:04X}  {len(f.payload):2d} B  seq={f.seq:3d}  {f.payload.hex()}")
    if len(frames) > 12:
        print(f"   …还有 {len(frames) - 12} 帧")


def main() -> None:
    p = argparse.ArgumentParser(description="PlutoSDR 接收探针")
    p.add_argument("--uri", default="", help="接收板 URI，留空自动选扫描到的第一台")
    p.add_argument("--team", default="red", choices=["red", "blue"])
    p.add_argument("--wave", default="info",
                   choices=["info", "jammer_1", "jammer_2", "jammer_3"])
    p.add_argument("--offset", type=float, default=-300e3,
                   help="RX 本振相对信号频点的偏移 Hz，默认 -300k（避开直流泄漏）")
    p.add_argument("--gain", type=float, default=40,
                   help="RX 硬件增益 dB（-3~71），默认 40；信号弱就往 50~60 加")
    p.add_argument("--seconds", type=float, default=5, help="监听时长，默认 5 秒")
    p.add_argument("--stop-after", type=int, default=20, help="解出多少帧后提前结束")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
