# -*- coding: utf-8 -*-
"""帧构造关键链路测试：业务帧回环、连续切片、跨包滑窗恢复、空口包结构。"""

import os

from radar_tx.protocol.air_frame import (
    AIR_FRAME_LEN, AIR_HEADER, AIR_PAYLOAD_LEN, ACCESS_CODE_INFO,
    ACCESS_CODE_INTERFERE, AirStreamChunker, build_air_packet,
    bytes_to_bits, extract_air_payloads,
)
from radar_tx.protocol.serial_frame import (
    CMD_INTERFERENCE_KEY, CMD_ROBOT_BUFF, CMD_ROBOT_POSITION,
    SerialFrameBuilder, SerialFrameDecoder,
)


def test_serial_frame_roundtrip():
    builder = SerialFrameBuilder()
    frames = [
        builder.pack_position(range(12)),
        builder.pack_key("AB12"),
    ]
    decoder = SerialFrameDecoder()
    decoded = decoder.feed(b"".join(frames))
    assert [f.cmd_id for f in decoded] == [CMD_ROBOT_POSITION, CMD_INTERFERENCE_KEY]
    assert decoded[1].payload == b"AB1200"  # 密钥不足 6 位右侧补 '0'


def test_buff_payload_is_47_bytes():
    # 0x0A05 载荷固定 47 字节（通信协议 V1.0.0 表 1-45）
    builder = SerialFrameBuilder()
    frame = builder.pack_buff(os.urandom(47))
    assert len(frame) == 47 + 9
    try:
        builder.pack_buff(os.urandom(41))
        assert False, "41 字节载荷应被拒绝"
    except ValueError:
        pass


def test_key_frame_is_exactly_one_air_payload():
    # 密钥帧 5+2+6+2 = 15 字节，恰好装入单个空口载荷
    builder = SerialFrameBuilder()
    assert len(builder.pack_key("ABCDEF")) == AIR_PAYLOAD_LEN


def test_stream_chunker_continuity():
    # 信息波 150B/周期恰好 10 个空口包；切片余量仍须跨周期延续
    chunker = AirStreamChunker(ACCESS_CODE_INFO)
    total_packets = 0
    for _ in range(3):
        total_packets += len(chunker.feed(os.urandom(150)))
    # 3 周期共 450B → 30 个空口包
    assert total_packets == 30


def test_cross_packet_sliding_decode():
    # 业务帧流跨空口包且混入随机填充（干扰波规则形态），滑窗仍能恢复
    builder = SerialFrameBuilder()
    key_frame = builder.pack_key("KEY001")
    stream = os.urandom(7) + key_frame + os.urandom(135 - 7 - len(key_frame))
    chunker = AirStreamChunker(ACCESS_CODE_INTERFERE)
    packets = chunker.feed(stream)
    payload_stream = b"".join(p[AIR_FRAME_LEN - AIR_PAYLOAD_LEN :] for p in packets)
    decoder = SerialFrameDecoder()
    decoded = decoder.feed(payload_stream)
    assert [f.cmd_id for f in decoded] == [CMD_INTERFERENCE_KEY]
    assert decoded[0].payload == b"KEY001"


def test_air_packet_structure_and_extraction():
    packet = build_air_packet(ACCESS_CODE_INFO, bytes(range(15)))
    assert len(packet) == AIR_FRAME_LEN
    assert packet[:8] == ACCESS_CODE_INFO
    assert packet[8:12] == AIR_HEADER
    bits = bytes_to_bits(packet)
    payloads = extract_air_payloads(bits, bytes_to_bits(ACCESS_CODE_INFO))
    assert payloads == [bytes(range(15))]
