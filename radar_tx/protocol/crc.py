# -*- coding: utf-8 -*-
"""RMUC 风格 CRC8 / CRC16 实现。

采用反射多项式：CRC8 为 0x8C（Dallas/Maxim），CRC16 为 0x8408（X.25 系），
初始值分别为 0xFF / 0xFFFF，与裁判系统串口协议一致。
查表由多项式在线生成，避免手工大表抄写错误；测试中与官方风格常量表对拍。
"""

from __future__ import annotations

# 反射多项式常量
_CRC8_POLY = 0x8C
_CRC16_POLY = 0x8408

CRC8_INIT = 0xFF
CRC16_INIT = 0xFFFF


def _build_crc8_table() -> tuple:
    """按反射多项式生成 CRC8 查表。"""
    table = []
    for byte in range(256):
        crc = byte
        for _ in range(8):
            crc = (crc >> 1) ^ _CRC8_POLY if (crc & 1) else (crc >> 1)
        table.append(crc & 0xFF)
    return tuple(table)


def _build_crc16_table() -> tuple:
    """按反射多项式生成 CRC16 查表。"""
    table = []
    for byte in range(256):
        crc = byte
        for _ in range(8):
            crc = (crc >> 1) ^ _CRC16_POLY if (crc & 1) else (crc >> 1)
        table.append(crc & 0xFFFF)
    return tuple(table)


CRC8_TABLE = _build_crc8_table()
CRC16_TABLE = _build_crc16_table()


def crc8(data: bytes, init: int = CRC8_INIT) -> int:
    """帧头 CRC8，覆盖 SOF + data_length + seq 共 4 字节。"""
    crc = init & 0xFF
    for byte in data:
        crc = CRC8_TABLE[(crc ^ byte) & 0xFF]
    return crc


def crc16(data: bytes, init: int = CRC16_INIT) -> int:
    """整帧 CRC16，覆盖 SOF 到 payload 结束（不含 CRC16 自身）。"""
    crc = init & 0xFFFF
    for byte in data:
        crc = ((crc >> 8) ^ CRC16_TABLE[(crc ^ byte) & 0xFF]) & 0xFFFF
    return crc
