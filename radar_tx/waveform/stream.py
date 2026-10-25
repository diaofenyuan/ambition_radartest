# -*- coding: utf-8 -*-
"""空口包序列 → 连续基带 IQ。"""

from __future__ import annotations

from typing import List

import numpy as np

from .gfsk import GfskModulator


def modulate_air_packets(packets: List[bytes], modulator: GfskModulator) -> np.ndarray:
    """整段拼接后一次成形，形成连续发射流（包间无保护间隔）。

    规则附录 1 明确"无固定包间隔，数据到达即发射"，所以包之间不能各自成形：
    逐包调制会在每个包边界把相位归零，瞬时频偏跳出 ±Δf 范围（实测可达
    -446 kHz，而设计频偏只有 248.7 kHz），并在包首尾各留下一个频偏减半的符号。
    调制器本身带跨块状态，连续调用即等效于一条不断的数据流。
    """
    if not packets:
        return np.zeros(0, dtype=np.complex64)
    return modulator.modulate_stream(b"".join(packets))
