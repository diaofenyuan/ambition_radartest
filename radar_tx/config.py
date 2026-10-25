# -*- coding: utf-8 -*-
"""赛季参数与设备配置加载。

设计要点：规则参数（频点/带宽/调制）与设备参数分离，赛季更替只换 YAML；
频偏不直接写死，由带宽按规则公式推导（f_dev = BW/2 - 码率），保证换赛季后自洽。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import yaml

ROOT = Path(__file__).resolve().parent.parent
RF_PROFILES_PATH = ROOT / "config" / "rf_profiles_2026.yaml"
DEVICES_PATH = ROOT / "config" / "devices.yaml"


@dataclass(frozen=True)
class RfProfile:
    """单条射频配置（频点/带宽/标称功率/码率）。"""
    freq_hz: int
    bandwidth_hz: int
    power_dbm: int
    # 码率由 ModulationParams 决定（采样率 ÷ SPS），随赛季一起写在 YAML 里。
    # 这里必须保存加载后的实参：早先版本直接读 ModulationParams 的类属性，
    # 等于永远取 dataclass 默认值（1MS/s ÷ 47），改 YAML 换赛季后频偏不会跟随，
    # 与"换赛季只换 YAML 不动代码"的设计目标相悖。
    bit_rate_hz: float

    @property
    def freq_deviation_hz(self) -> float:
        """2-GFSK 频偏：带宽半宽减去码率，对齐规则表 5-23 的 sensitivity 推导。"""
        return self.bandwidth_hz / 2 - self.bit_rate_hz


@dataclass(frozen=True)
class ModulationParams:
    SAMPLE_RATE_HZ: int = 1_000_000
    SAMPLES_PER_SYMBOL: int = 47
    GFSK_BT: float = 0.35
    GAUSSIAN_SPAN_SYMBOLS: int = 4

    @property
    def bit_rate_hz(self) -> float:
        return self.SAMPLE_RATE_HZ / self.SAMPLES_PER_SYMBOL


@dataclass(frozen=True)
class SeasonConfig:
    """赛季完整参数集。"""
    season: int
    modulation: ModulationParams
    access_code_info: bytes
    access_code_interfere: bytes
    cycles_per_sec: int
    key_len: int
    key_pad_char: str
    buff_payload_lens: tuple
    info_bytes_per_cycle: int
    jammer_bytes_per_cycle: int
    profiles: Dict[str, Dict[str, RfProfile]]  # profiles[team][wave]
    tx_limits: dict

    def profile(self, team: str, wave: str) -> RfProfile:
        """wave: info / jammer_1 / jammer_2 / jammer_3"""
        return self.profiles[team][wave]


def load_season_config(path: Path = RF_PROFILES_PATH) -> SeasonConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    mod_raw = raw["modulation"]
    modulation = ModulationParams(
        SAMPLE_RATE_HZ=int(mod_raw["sample_rate_hz"]),
        SAMPLES_PER_SYMBOL=int(mod_raw["samples_per_symbol"]),
        GFSK_BT=float(mod_raw["gfsk_bt"]),
        GAUSSIAN_SPAN_SYMBOLS=int(mod_raw["gaussian_span_symbols"]),
    )
    air = raw["air_protocol"]
    biz = raw["business"]
    profiles: Dict[str, Dict[str, RfProfile]] = {}
    for team, waves in raw["rf_profiles"].items():
        profiles[team] = {
            wave: RfProfile(
                freq_hz=int(spec["freq_hz"]),
                bandwidth_hz=int(spec["bandwidth_hz"]),
                power_dbm=int(spec["power_dbm"]),
                bit_rate_hz=modulation.bit_rate_hz,
            )
            for wave, spec in waves.items()
        }
    return SeasonConfig(
        season=int(raw["season"]),
        modulation=modulation,
        access_code_info=bytes.fromhex(air["access_code_info"]),
        access_code_interfere=bytes.fromhex(air["access_code_interfere"]),
        cycles_per_sec=int(air["cycles_per_sec"]),
        key_len=int(biz["key_len_ascii"]),
        key_pad_char=str(biz["key_pad_char"]),
        buff_payload_lens=tuple(biz["buff_payload_lens"]),
        info_bytes_per_cycle=int(biz["info_bytes_per_cycle"]),
        jammer_bytes_per_cycle=int(biz["jammer_bytes_per_cycle"]),
        profiles=profiles,
        tx_limits=dict(raw["tx_limits"]),
    )


@dataclass
class DeviceConfig:
    """单块发射板配置。"""
    name: str
    uri: str
    attenuation_db: float


@dataclass
class DevicesConfig:
    mode: str  # sim / pluto
    info_tx: DeviceConfig
    jammer_tx: DeviceConfig
    tx_buffer_samples: int = 32768
    underflow_gap_ms: float = 15.0


def load_devices_config(path: Path = DEVICES_PATH) -> DevicesConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    stream = raw.get("stream", {})
    return DevicesConfig(
        mode=str(raw.get("mode", "sim")),
        info_tx=DeviceConfig(**raw["info_tx"]),
        jammer_tx=DeviceConfig(**raw["jammer_tx"]),
        tx_buffer_samples=int(stream.get("tx_buffer_samples", 32768)),
        underflow_gap_ms=float(stream.get("underflow_gap_ms", 15)),
    )
