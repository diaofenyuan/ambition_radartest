# -*- coding: utf-8 -*-
"""空口帧封装与连续包流构造（规则手册 V2.2.0 附录 1 的 2026 口径）。

2027 通信协议 V1.0.0 只定义业务帧（§1.5），未给出空口帧结构/比特率；
2027 规则手册发布前沿用本组参数，服务启动会核算业务字节是否仍装得下周期。

空口帧固定 27 字节：Access Code(8) + Header(4，两份 Payload Length=15 的大端副本) + Payload(15)。
发送的数据流按 15 字节为一组连续切片成空口包，无固定包间隔（数据到达即发射），
切片跨周期持续，余量留到下周期——AirStreamChunker 维护该跨周期余量。
"""

from __future__ import annotations

from typing import List

# 空口同步字，大端序；红蓝双方共用同一对 Access Code（规则附录 1）
ACCESS_CODE_INFO = bytes.fromhex("2F6F4C74B914492E")
ACCESS_CODE_INTERFERE = bytes.fromhex("16E8D377151C712D")

AIR_HEADER = bytes.fromhex("000F000F")
AIR_PAYLOAD_LEN = 15
AIR_FRAME_LEN = 8 + 4 + AIR_PAYLOAD_LEN  # 27
AIR_CYCLES_PER_SEC = 10


def build_air_packet(access_code: bytes, payload: bytes) -> bytes:
    """构造单个 27 字节空口包。"""
    if len(access_code) != 8:
        raise ValueError("Access Code 必须为 8 字节")
    if len(payload) != AIR_PAYLOAD_LEN:
        raise ValueError(f"空口载荷必须为 {AIR_PAYLOAD_LEN} 字节")
    return access_code + AIR_HEADER + payload


class AirStreamChunker:
    """连续业务字节流 → 空口包序列（15 字节持续切片，跨周期保留余量）。

    规则为恒定字节速率（信息波 1500B/s、干扰波 1350B/s）。当前两者恰好都是
    15 的整数倍（10 包 / 9 包），但切片器不假设整除：余量跨周期保留，
    换赛季改变字节速率时也不会让接收侧看到周期性错位。
    """

    def __init__(self, access_code: bytes) -> None:
        self._access_code = access_code
        self._carry = bytearray()

    def feed(self, data: bytes) -> List[bytes]:
        self._carry.extend(data)
        packets: List[bytes] = []
        while len(self._carry) >= AIR_PAYLOAD_LEN:
            chunk = bytes(self._carry[:AIR_PAYLOAD_LEN])
            del self._carry[:AIR_PAYLOAD_LEN]
            packets.append(build_air_packet(self._access_code, chunk))
        return packets


def extract_air_payloads(bits: List[int], access_code: List[int], max_bit_errors: int = 0) -> List[bytes]:
    """从判决比特流中按 Access Code 提取空口载荷（回环自检用）。

    仿真链路无噪声，允许 max_bit_errors=0 严格匹配；
    保留容错参数以便后续接入真实射频样本。
    """
    code_len = len(access_code)
    frame_bits = AIR_FRAME_LEN * 8
    payloads: List[bytes] = []
    i = 0
    n = len(bits)
    while i + frame_bits <= n:
        errors = 0
        for k in range(code_len):
            if bits[i + k] != access_code[k]:
                errors += 1
                if errors > max_bit_errors:
                    break
        if errors <= max_bit_errors:
            # 校验固定 Header，逐字节严格一致
            header_bits = bits[i + code_len : i + code_len + 32]
            header = _bits_to_bytes(header_bits)
            if header == AIR_HEADER:
                payload_bits = bits[i + code_len + 32 : i + frame_bits]
                payloads.append(_bits_to_bytes(payload_bits))
                i += frame_bits
                continue
        i += 1
    return payloads


def bytes_to_bits(data: bytes) -> List[int]:
    """字节流转比特序列（MSB 先发）。"""
    bits: List[int] = []
    for byte in data:
        for shift in range(7, -1, -1):
            bits.append((byte >> shift) & 1)
    return bits


def _bits_to_bytes(bits: List[int]) -> bytes:
    out = bytearray()
    for off in range(0, len(bits) - 7, 8):
        byte = 0
        for shift in range(8):
            byte = (byte << 1) | bits[off + shift]
        out.append(byte)
    return bytes(out)
