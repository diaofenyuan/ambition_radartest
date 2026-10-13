# -*- coding: utf-8 -*-
"""空口包序列 → 连续基带 IQ 流。"""

from __future__ import annotations

from typing import List

import numpy as np

from .gfsk import GfskModulator


def modulate_air_packets(packets: List[bytes], modulator: GfskModulator) -> np.ndarray:
    """逐包调制后首尾拼接，形成连续发射流（包间无保护间隔）。"""
    if not packets:
        return np.zeros(0, dtype=np.complex64)
    return np.concatenate([modulator.modulate_bytes(p) for p in packets])
