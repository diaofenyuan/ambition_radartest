# -*- coding: utf-8 -*-
"""libiio 原生库定位与加载准备。

pylibiio 只是一层 ctypes 外壳，本身不含原生库。Windows 下它在导入时执行
`ctypes.util.find_library("libiio.dll")`，**只在 PATH 里查找**，找不到就
`CDLL(None)` 抛 TypeError（注意：不是 ImportError，日志里只显示
"argument of type 'NoneType' is not iterable"，很难看出是缺 DLL）。

因此凡是可能 `import iio` / `import adi` 的路径，都必须先调用
`ensure_libiio_available()`。否则真机模式会表现为"扫描不到任何设备"，
而仿真模式又一切正常，故障点被完全掩盖。

版本约束：pylibiio 0.25 对应 libiio 0.25 的 ABI，libiio 1.x 改了符号名，
不能混用（requirements.txt 已把绑定锁在 <1.0）。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# 随仓库分发的原生库目录（vendor/libiio 由 start.bat 首次运行时解压或手动放置）
_VENDOR_DIR = Path(__file__).resolve().parents[2] / "vendor" / "libiio"
_DLL_NAME = "libiio.dll"

# 保持 add_dll_directory 返回的句柄存活：句柄被回收时目录会从搜索路径中移除
_dll_handles: list = []
_ready_dir: Path | None = None


def candidate_dirs() -> list[Path]:
    """按优先级列出可能的 libiio 目录。"""
    dirs: list[Path] = []
    env = os.environ.get("LIBIIO_PATH")
    if env:
        # 允许指到 dll 本身或它所在的目录
        p = Path(env)
        dirs.append(p if p.is_dir() else p.parent)
    dirs.append(_VENDOR_DIR)
    # 官方 Windows 安装包（libiio-*-setup.exe）的默认落点
    program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
    dirs.append(Path(program_files) / "libiio" / "lib")
    dirs.append(Path(program_files) / "libiio" / "bin")
    return [d for d in dirs if d.is_dir()]


def find_libiio_dir() -> Path | None:
    """返回第一个含 libiio.dll 的目录。"""
    for d in candidate_dirs():
        if (d / _DLL_NAME).is_file():
            return d
    return None


def ensure_libiio_available() -> tuple[bool, str]:
    """把 libiio 目录挂进 DLL 搜索路径。

    返回 (是否可用, 说明)。说明在成功时是目录路径，失败时是给人看的排查提示。
    """
    global _ready_dir
    if sys.platform != "win32":
        # Linux/macOS 由系统包管理器安装，find_library 能自己找到
        return True, ""
    if _ready_dir is not None:
        return True, str(_ready_dir)

    d = find_libiio_dir()
    if d is None:
        hint = (
            "未找到 libiio.dll：pluto 模式需要 libiio 0.25 原生库。"
            "把它放到 vendor/libiio/，或用 LIBIIO_PATH 指向该目录。"
            "（仿真模式不受影响）"
        )
        return False, hint

    # find_library 只认 PATH，因此必须先改环境变量；
    # add_dll_directory 负责 libusb/libxml2 等依赖 DLL 的解析。
    os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
    try:
        _dll_handles.append(os.add_dll_directory(str(d)))
    except (AttributeError, OSError):
        pass
    _ready_dir = d
    return True, str(d)


def unavailable_reason() -> str:
    """给界面用的一句话诊断：为什么扫不到设备。"""
    ok, detail = ensure_libiio_available()
    if ok:
        return "libiio 已加载，但未枚举到 PlutoSDR（检查 USB 线/供电/驱动）"
    return detail
