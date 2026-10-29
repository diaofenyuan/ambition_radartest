# -*- coding: utf-8 -*-
"""RMUC 串口业务帧构造与解析（0x0A01 ~ 0x0A06）。

帧格式：SOF(0xA5) + data_length(u16 LE) + seq(u8) + CRC8 + cmd_id(u16 LE)
        + payload + CRC16(u16 LE)。
发射侧按连续字节流切进空口包；接收侧用滑窗恢复（一个业务帧可跨多个空口包），
因此本模块同时提供构造器与滑窗解码器，供仿真回环自检使用。
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

from .crc import CRC8_INIT, CRC16_INIT, crc8, crc16

SOF = 0xA5

# 业务命令 ID（2027 通信协议 V1.0.0 §1.5 表 1-41 ~ 1-46）
CMD_ROBOT_POSITION = 0x0A01  # 敌方机器人坐标，12 × int16（x,y 单位 cm，第 2/5 组为保留位）
CMD_ROBOT_HP = 0x0A02        # 6 × uint16，第 2/5 项为保留位
CMD_ROBOT_AMMO = 0x0A03      # 5 × uint16，第 4 项为保留位
CMD_FIELD_STATUS = 0x0A04    # 宏观状态：剩余/总金币 + 两个 4 字节状态字，共 12 字节
CMD_ROBOT_BUFF = 0x0A05      # 增益、状态与装甲加固圆，固定 47 字节（表 1-45）
CMD_INTERFERENCE_KEY = 0x0A06  # 干扰密钥，6 字节 ASCII

ALL_CMDS = (
    CMD_ROBOT_POSITION, CMD_ROBOT_HP, CMD_ROBOT_AMMO,
    CMD_FIELD_STATUS, CMD_ROBOT_BUFF, CMD_INTERFERENCE_KEY,
)


@dataclass(frozen=True)
class SerialFrame:
    """已通过 CRC 校验的业务帧。"""
    cmd_id: int
    payload: bytes
    seq: int

    @property
    def raw(self) -> bytes:
        return build_serial_frame(self.cmd_id, self.payload, self.seq)


def build_serial_frame(cmd_id: int, payload: bytes, seq: int) -> bytes:
    """构造完整业务帧（含 CRC8 / CRC16）。"""
    header_no_crc8 = struct.pack("<BHB", SOF, len(payload), seq & 0xFF)
    body = struct.pack("<H", cmd_id & 0xFFFF) + payload
    head_crc = crc8(header_no_crc8, CRC8_INIT)
    frame_wo_crc16 = header_no_crc8 + bytes([head_crc]) + body
    return frame_wo_crc16 + struct.pack("<H", crc16(frame_wo_crc16, CRC16_INIT))


class SerialFrameBuilder:
    """业务帧构造器，seq 自增覆盖 0~255。"""

    def __init__(self) -> None:
        self._seq = 0

    def pack(self, cmd_id: int, payload: bytes) -> bytes:
        frame = build_serial_frame(cmd_id, payload, self._seq)
        self._seq = (self._seq + 1) & 0xFF
        return frame

    # ---- 语义化业务载荷构造（字段定义对齐通信协议 V1.0.0 §1.5） ----

    def pack_position(self, coords_cm: Iterable[int]) -> bytes:
        """coords_cm 为 12 个 int16（6 组 x,y）；第 2、5 组按表 1-41 为保留位。"""
        values = list(coords_cm)
        if len(values) != 12:
            raise ValueError("坐标必须为 12 个 int16（6 台 × x,y）")
        return self.pack(CMD_ROBOT_POSITION, struct.pack("<12h", *values))

    def pack_hp(self, hps: Iterable[int]) -> bytes:
        values = list(hps)
        if len(values) != 6:
            raise ValueError("血量必须为 6 个 uint16")
        return self.pack(CMD_ROBOT_HP, struct.pack("<6H", *values))

    def pack_ammo(self, ammos: Iterable[int]) -> bytes:
        values = list(ammos)
        if len(values) != 5:
            raise ValueError("弹量必须为 5 个 uint16")
        return self.pack(CMD_ROBOT_AMMO, struct.pack("<5H", *values))

    def pack_field_status(self, remain_coins: int, total_coins: int,
                          gauge_flags: int, terrain_flags: int) -> bytes:
        """0x0A04：金币 + 增益点状态字 + 地形跨越/阵地控制状态字（表 1-44）。"""
        return self.pack(
            CMD_FIELD_STATUS,
            struct.pack("<HHII", remain_coins & 0xFFFF, total_coins & 0xFFFF,
                        gauge_flags & 0xFFFFFFFF, terrain_flags & 0xFFFFFFFF),
        )

    def pack_buff(self, payload: bytes) -> bytes:
        """0x0A05 载荷固定 47 字节（表 1-45：5 × 7B 增益 + 6B 状态 + 6B 加固圆）。"""
        if len(payload) != 47:
            raise ValueError("0x0A05 载荷必须为 47 字节")
        return self.pack(CMD_ROBOT_BUFF, payload)

    def pack_key(self, key_text: str, pad_char: str = "0") -> bytes:
        """干扰密钥固定 6 字节 ASCII（表 1-46），不足右侧补 pad_char。

        补位只是兜底：正常路径上密钥在状态层就已补齐到 6 位，
        这样界面显示的串与真正发出去的那 6 个字节才是同一个东西。
        """
        pad = (pad_char or "0")[:1].encode("ascii") or b"0"
        raw = key_text[:6].encode("ascii", errors="ignore").ljust(6, pad)
        return self.pack(CMD_INTERFERENCE_KEY, raw)


class SerialFrameDecoder:
    """连续字节流滑窗解码器。

    空口载荷中混有随机填充，且业务帧可能跨空口包，因此不能假设 SOF 对齐；
    解码失败时从 SOF 后一字节继续搜索，实现自动重同步。
    """

    def __init__(self, max_payload_len: int = 200) -> None:
        self._buf = bytearray()
        self._max_payload_len = max_payload_len

    def feed(self, data: bytes) -> List[SerialFrame]:
        self._buf.extend(data)
        frames: List[SerialFrame] = []
        while True:
            frame, consumed = self._try_extract()
            if consumed == 0:
                break
            del self._buf[:consumed]
            if frame is not None:
                frames.append(frame)
        return frames

    def _try_extract(self) -> Tuple[Optional[SerialFrame], int]:
        buf = self._buf
        # 1. 定位 SOF
        try:
            start = buf.index(SOF)
        except ValueError:
            return None, len(buf)
        if start > 0:
            return None, start  # 丢弃 SOF 前的垃圾字节
        if len(buf) < 5:
            return None, 0  # 等待帧头到齐
        # 2. 校验帧头 CRC8
        data_len = struct.unpack_from("<H", buf, 1)[0]
        if crc8(bytes(buf[:4]), CRC8_INIT) != buf[4]:
            return None, 1  # 帧头无效，SOF 后移一字节继续搜
        if data_len > self._max_payload_len:
            return None, 1
        # 3. 等待完整帧：帧头 5B + cmd 2B + payload + CRC16 2B
        total = 5 + 2 + data_len + 2
        if len(buf) < total:
            return None, 0
        raw = bytes(buf[:total])
        body_wo_crc16 = raw[:-2]
        if crc16(body_wo_crc16, CRC16_INIT) != struct.unpack_from("<H", raw, total - 2)[0]:
            return None, 1  # 整帧 CRC 失败
        cmd_id = struct.unpack_from("<H", raw, 5)[0]
        payload = raw[7 : 7 + data_len]
        return SerialFrame(cmd_id=cmd_id, payload=payload, seq=raw[3]), total
