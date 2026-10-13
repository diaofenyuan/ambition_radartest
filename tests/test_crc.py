# -*- coding: utf-8 -*-
"""CRC 关键链路测试：查表实现与逐位参考实现对拍，并核对官方风格常量表抽样值。"""

import random

from radar_tx.protocol.crc import CRC8_TABLE, CRC16_TABLE, crc8, crc16


def _crc8_bitwise(data: bytes, init: int = 0xFF) -> int:
    """逐位参考实现（反射多项式 0x8C），用于对拍查表结果。"""
    crc = init
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0x8C if (crc & 1) else (crc >> 1)
    return crc & 0xFF


def _crc16_bitwise(data: bytes, init: int = 0xFFFF) -> int:
    """逐位参考实现（反射多项式 0x8408）。"""
    crc = init
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0x8408 if (crc & 1) else (crc >> 1)
    return crc & 0xFFFF


def test_crc8_table_matches_official_samples():
    # 与裁判系统风格 CRC8 表（反射多项式 0x8C）的抽样对拍
    assert CRC8_TABLE[0x00] == 0x00
    assert CRC8_TABLE[0x01] == 0x5E
    assert CRC8_TABLE[0x02] == 0xBC
    assert CRC8_TABLE[0x0F] == 0x41
    assert CRC8_TABLE[0xFF] == 0x35


def test_crc16_table_matches_official_samples():
    # 与裁判系统风格 CRC16 表（反射多项式 0x8408）的抽样对拍
    assert CRC16_TABLE[0x0000 & 0xFF] == 0x0000
    assert CRC16_TABLE[0x01] == 0x1189
    assert CRC16_TABLE[0x02] == 0x2312
    assert CRC16_TABLE[0x0F] == 0xF8F7
    assert CRC16_TABLE[0xFF] == 0x0F78


def test_crc_lookup_equals_bitwise():
    rng = random.Random(2026)
    for _ in range(50):
        data = bytes(rng.randrange(256) for _ in range(rng.randrange(1, 128)))
        assert crc8(data) == _crc8_bitwise(data)
        assert crc16(data) == _crc16_bitwise(data)
