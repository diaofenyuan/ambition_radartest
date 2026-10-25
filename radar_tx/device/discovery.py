# -*- coding: utf-8 -*-
"""PlutoSDR 设备发现。

libiio 的 ip/usb 后端会把同一块板同时枚举成多个上下文（USB 直连、mDNS），
因此必须按序列号去重；优先保留 USB 上下文，不依赖网络配置和 mDNS 解析。
扫描失败按"无设备"处理，不能影响仿真模式启动。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict

_SERIAL_RE = re.compile(r"serial=([0-9A-Za-z]+)")


@dataclass(frozen=True)
class DiscoveredDevice:
    """一次扫描得到的 PlutoSDR 逻辑设备。"""
    uri: str
    description: str
    serial: str
    model: str


def _prefer_rank(uri: str) -> int:
    """同一块板有多个 URI 时的优先级：USB > IP > 其他。"""
    if uri.startswith("usb:"):
        return 0
    if uri.startswith("ip:"):
        return 1
    return 2


def _is_pluto(uri: str, description: str) -> bool:
    """过滤非 Pluto 的 IIO 上下文，避免把其他设备误分配给发射波源。"""
    text = f"{uri} {description}".lower()
    return "pluto" in text or "0456:b673" in text


def _model_name(description: str) -> str:
    if "Rev.B" in description:
        return "PlutoSDR Rev.B"
    if "Rev.A" in description:
        return "PlutoSDR Rev.A"
    return "PlutoSDR"


def scan_pluto_devices() -> list[DiscoveredDevice]:
    """扫描当前主机可见的 PlutoSDR；无依赖或扫描失败时返回空列表。"""
    try:
        import iio  # 延迟导入：仿真环境无需安装 libiio 绑定
    except ImportError:
        return []
    try:
        contexts = iio.scan_contexts() or {}
    except Exception:
        return []

    found: Dict[str, DiscoveredDevice] = {}
    for uri, description in contexts.items():
        description = str(description)
        if not _is_pluto(str(uri), description):
            continue
        match = _SERIAL_RE.search(description)
        serial = match.group(1) if match else ""
        # 没有序列号时按 URI 去重，避免把不同设备合并
        key = serial or str(uri)
        candidate = DiscoveredDevice(
            uri=str(uri),
            description=description,
            serial=serial,
            model=_model_name(description),
        )
        old = found.get(key)
        if old is None or _prefer_rank(candidate.uri) < _prefer_rank(old.uri):
            found[key] = candidate

    return sorted(found.values(), key=lambda d: (d.serial, d.uri))
