# -*- coding: utf-8 -*-
"""协议层：CRC、业务帧、空口帧。"""

from .crc import crc8, crc16, CRC8_TABLE, CRC16_TABLE
from .serial_frame import (
    SerialFrame, SerialFrameBuilder, SerialFrameDecoder, build_serial_frame,
    CMD_ROBOT_POSITION, CMD_ROBOT_HP, CMD_ROBOT_AMMO,
    CMD_FIELD_STATUS, CMD_ROBOT_BUFF, CMD_INTERFERENCE_KEY, ALL_CMDS,
)
from .air_frame import (
    ACCESS_CODE_INFO, ACCESS_CODE_INTERFERE, AIR_HEADER, AIR_PAYLOAD_LEN,
    AIR_FRAME_LEN, AirStreamChunker,
    build_air_packet, extract_air_payloads, bytes_to_bits,
)
