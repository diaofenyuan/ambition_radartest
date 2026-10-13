# -*- coding: utf-8 -*-
"""无硬件回环自检：基带 IQ → 空口包 → 业务帧。

用途：在没有 SDR 的开发/演示环境下验证"业务数据 → 帧 → 波形"发射链路正确性。
链路与真实接收侧共享同一份协议常量，但解调仅做仿真级简化（见 gfsk 解调器说明）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from ..config import SeasonConfig
from ..protocol.air_frame import bytes_to_bits, extract_air_payloads
from ..protocol.serial_frame import SerialFrame, SerialFrameDecoder
from ..waveform.gfsk import GfskDemodulator


@dataclass
class LoopbackResult:
    """单次回环解码结果。"""
    air_payloads: List[bytes] = field(default_factory=list)
    frames: List[SerialFrame] = field(default_factory=list)

    @property
    def cmd_ids(self) -> List[int]:
        return [f.cmd_id for f in self.frames]


class LoopbackDecoder:
    def __init__(self, season: SeasonConfig) -> None:
        self.season = season
        self._demod = GfskDemodulator(season.modulation)

    def decode(self, iq: np.ndarray, is_info: bool) -> LoopbackResult:
        access_code = self.season.access_code_info if is_info else self.season.access_code_interfere
        code_bits = bytes_to_bits(access_code)
        bits = self._demod.demodulate_to_bits(iq, code_bits)
        bit_list = bits.tolist()
        payloads = extract_air_payloads(bit_list, code_bits, max_bit_errors=0)
        decoder = SerialFrameDecoder()
        frames: List[SerialFrame] = []
        for payload in payloads:
            frames.extend(decoder.feed(payload))
        return LoopbackResult(air_payloads=payloads, frames=frames)
