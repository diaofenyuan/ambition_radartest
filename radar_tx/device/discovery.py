# -*- coding: utf-8 -*-
"""PlutoSDR 设备发现。

libiio 的 ip/usb 后端会把同一块板同时枚举成多个上下文（USB 直连、mDNS），
因此必须按序列号去重；优先保留 USB 上下文，不依赖网络配置和 mDNS 解析。
扫描失败按"无设备"处理，不能影响仿真模式启动。
"""

from __future__ import annotations

import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict

from .libiio_runtime import ensure_libiio_available

_SERIAL_RE = re.compile(r"serial=([0-9A-Za-z]+)")
_IPV4_RE = re.compile(r"\d{1,3}(?:\.\d{1,3}){3}")

# 最近一次扫描的可读说明：扫不到设备时界面用它解释原因，空串表示正常
_LAST_NOTE = ""


def last_scan_note() -> str:
    return _LAST_NOTE


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


def _needs_reachability_probe(uri: str) -> bool:
    """只有 ip:<主机名> 这种 URI 需要实测。

    libiio 会把同一块板通过 mDNS 广播成一个 ip:pluto.local 上下文。机器上没有
    mDNS 解析器时这个名字会被普通 DNS 抢答——实测在本机解析到公网 198.20.0.21，
    照抄它去绑定设备必然连不上。与其给出一个连不上的地址，不如不给。
    纯数字 IP 与 USB 上下文直接信任，免得每次扫描都多等两秒。
    """
    if not uri.startswith("ip:"):
        return False
    host = uri[3:].split(":")[0]
    return _IPV4_RE.fullmatch(host) is None


def _reachable(uri: str, timeout_s: float = 2.0) -> bool:
    """建立一次真实上下文确认 URI 可用；超时按不可用处理。"""
    import iio
    verdict: Dict[str, bool] = {}

    def probe() -> None:
        try:
            iio.Context(uri)
            verdict["ok"] = True
        except Exception:
            verdict["ok"] = False

    worker = threading.Thread(target=probe, daemon=True)
    worker.start()
    worker.join(timeout_s)
    return verdict.get("ok", False)


def scan_pluto_devices() -> list[DiscoveredDevice]:
    """扫描当前主机可见的 PlutoSDR；无依赖或扫描失败时返回空列表。"""
    global _LAST_NOTE
    _LAST_NOTE = ""
    # 必须在 import iio 之前把原生库目录挂进 PATH，否则 find_library 找不到
    # libiio.dll，导入阶段直接抛 TypeError。
    libiio_ok, _ = ensure_libiio_available()
    if not libiio_ok:
        return []
    try:
        import iio  # 延迟导入：仿真环境无需安装 libiio 绑定
    except Exception:
        # 兜底：原生库存在但依赖 DLL 缺失时，导入仍可能抛非 ImportError。
        return []
    try:
        contexts = iio.scan_contexts() or {}
    except Exception:
        return []

    candidates: list[tuple[str, str]] = []
    for uri, description in contexts.items():
        uri = str(uri)
        description = str(description)
        if not _is_pluto(uri, description):
            continue
        candidates.append((uri, description))

    # 并行探测：ip:<主机名> 走到公网 DNS 时会一直卡到超时，串行时每块板叠 2 秒，
    # 两块板（各自还带一条 mDNS 上下文）就能把启动拖到 8 秒以上——比 start.bat
    # 的等待时间长，用户打开页面时服务还没起来。并起来只花一个超时周期。
    need_probe = [u for u, _ in candidates if _needs_reachability_probe(u)]
    verdict: Dict[str, bool] = {}
    if need_probe:
        with ThreadPoolExecutor(max_workers=min(8, len(need_probe))) as pool:
            for target, reachable in zip(need_probe, pool.map(_reachable, need_probe)):
                verdict[target] = reachable

    found: Dict[str, DiscoveredDevice] = {}
    skipped: list[str] = []
    for uri, description in candidates:
        if verdict.get(uri) is False:
            skipped.append(uri)
            continue
        match = _SERIAL_RE.search(description)
        serial = match.group(1) if match else ""
        # 没有序列号时按 URI 去重，避免把不同设备合并
        key = serial or uri
        candidate = DiscoveredDevice(
            uri=uri,
            description=description,
            serial=serial,
            model=_model_name(description),
        )
        old = found.get(key)
        if old is None or _prefer_rank(candidate.uri) < _prefer_rank(old.uri):
            found[key] = candidate

    if skipped and not found:
        _LAST_NOTE = (f"已忽略连不上的地址 {'、'.join(skipped)}；"
                      f"请改用 USB 直连，或手动填写 ip:192.168.2.1")
    return sorted(found.values(), key=lambda d: (d.serial, d.uri))
