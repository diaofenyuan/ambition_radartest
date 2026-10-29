# -*- coding: utf-8 -*-
"""发射调度服务：双波形状态机 + 帧生成 + 设备推送 + 仿真回环自检。

设计要点：
- 状态与执行分离：Web 控制只改内存状态（加锁），发射线程每周期取快照生成帧，
  因此阵营/等级/密钥/载荷均可热切换而不停发（对齐江南 LabTX 行为）；
- 射频安全门：未勾选安全确认不得启动任何发射，MUTE 同时停发、拉衰减并撤销确认；
- sim / pluto 后端可互换，仿真模式走同一套帧与波形代码，仅设备层不同。
"""

from __future__ import annotations

import math
import os
import random
import struct
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

from ..config import DevicesConfig, SeasonConfig
from ..device.base import TxBackend
from ..device.discovery import DiscoveredDevice, last_scan_note, scan_pluto_devices
from ..device.libiio_runtime import unavailable_reason as libiio_unavailable_reason
from ..device.pluto_tx import PlutoTx
from ..device.virtual_sdr import VirtualSdr
from ..protocol.air_frame import AIR_FRAME_LEN, AIR_PAYLOAD_LEN, AirStreamChunker
from ..protocol.serial_frame import (
    ALL_CMDS, CMD_FIELD_STATUS, CMD_INTERFERENCE_KEY, CMD_ROBOT_AMMO,
    CMD_ROBOT_BUFF, CMD_ROBOT_HP, CMD_ROBOT_POSITION, SerialFrameBuilder,
)
from ..sim.loopback import LoopbackDecoder
from ..waveform.gfsk import GfskModulator
from ..waveform.stream import modulate_air_packets

CMD_NAMES = {
    CMD_ROBOT_POSITION: "机器人位置",
    CMD_ROBOT_HP: "机器人血量",
    CMD_ROBOT_AMMO: "机器人弹量",
    CMD_FIELD_STATUS: "场地状态",
    CMD_ROBOT_BUFF: "增益/姿态",
    CMD_INTERFERENCE_KEY: "干扰密钥",
}
INFO_CMDS = (CMD_ROBOT_POSITION, CMD_ROBOT_HP, CMD_ROBOT_AMMO, CMD_FIELD_STATUS, CMD_ROBOT_BUFF)

# 各命令业务载荷字节数（2027 通信协议 V1.0.0 表 1-41 ~ 1-46）
CMD_PAYLOAD_LEN = {
    CMD_ROBOT_POSITION: 24, CMD_ROBOT_HP: 12, CMD_ROBOT_AMMO: 10,
    CMD_FIELD_STATUS: 12, CMD_ROBOT_BUFF: 47, CMD_INTERFERENCE_KEY: 6,
}

# 密钥字符集：规则为"6 位数字字母组合"
_KEY_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# 规则表 5-23 波源名称（用于 Web 规则核对面板）
WAVE_LABELS = {
    ("red", "info"): "红方广播源", ("red", "jammer_1"): "红方一级干扰源",
    ("red", "jammer_2"): "红方二级干扰源", ("red", "jammer_3"): "红方三级干扰源",
    ("blue", "info"): "蓝方广播源", ("blue", "jammer_1"): "蓝方一级干扰源",
    ("blue", "jammer_2"): "蓝方二级干扰源", ("blue", "jammer_3"): "蓝方三级干扰源",
}

# 敌方场地坐标范围（cm），用于自动变化载荷的随机游走
_FIELD_X_CM, _FIELD_Y_CM = 2800, 1500


@dataclass
class CommandState:
    """单条业务命令的发射状态。"""
    cmd_id: int
    enabled: bool = True
    payload_hex: str = ""
    auto_vary: bool = True


class TxService:
    """模拟发射系统总控。"""

    def __init__(self, season: SeasonConfig, devices: DevicesConfig,
                 device_scanner: Callable[[], List[DiscoveredDevice]] | None = None) -> None:
        self.season = season
        self.devices = devices
        self._device_scanner = device_scanner or scan_pluto_devices
        self._discovered_devices: List[DiscoveredDevice] = []
        # 波源到物理设备的动态绑定：为空时使用虚拟 SDR。
        # 这样同一块板可以在界面上被指定为信息波或干扰波，而不必改配置文件。
        self._assignments: Dict[str, Optional[str]] = {"info": None, "jammer": None}
        self._lock = threading.RLock()
        self._logs: Deque[str] = deque(maxlen=200)
        self._stop = threading.Event()
        self._threads: Dict[str, threading.Thread] = {}
        # 每次 start/stop 递增一次；发射线程持有自己的 run_id，
        # 与当前值不一致即退出。否则"停发后立刻再开"会让上一轮线程
        # 因 running 又被置位而继续发射，形成双线程并发推流。
        self._run_ids: Dict[str, int] = {"info": 0, "jammer": 0}

        # ---- 可被 Web 修改的运行状态 ----
        self.team = "red"
        self.safety_confirmed = False
        self.auto_vary = True
        self.jammer_level = 1
        # 规则：开局双方干扰波默认一级难度，密钥重置为随机值
        self.key = self._random_key().decode("ascii")
        self.info_running = False
        self.jammer_running = False
        self.attenuation = {
            "info": float(devices.info_tx.attenuation_db),
            "jammer": float(devices.jammer_tx.attenuation_db),
        }
        self.cmds: Dict[int, CommandState] = {
            cmd: CommandState(cmd_id=cmd) for cmd in INFO_CMDS
        }
        self._coords = [random.randint(0, _FIELD_X_CM) if i % 2 == 0 else random.randint(0, _FIELD_Y_CM)
                        for i in range(12)]
        self._zero_reserved_position()

        # ---- 发射链路 ----
        self._mods: Dict[Tuple[str, str], GfskModulator] = {}
        self._loopback = LoopbackDecoder(season)
        self._last_loopback: Dict[str, dict] = {"info": {}, "jammer": {}}
        # 回环解码线程池：解码必须离开发射线程，否则会拖慢仿真节拍（见 _make_backend）
        self._decode_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="loopback")
        self._discovered_devices = self._read_devices()
        if self.devices.mode == "pluto":
            # 兼容原有配置文件模式：启动即按配置 URI 绑定，仍可在界面上改绑扫描到的设备
            self._assignments = {
                "info": self.devices.info_tx.uri,
                "jammer": self.devices.jammer_tx.uri,
            }
        self.backends: Dict[str, TxBackend] = {
            "info": self._make_backend("info"),
            "jammer": self._make_backend("jammer"),
        }
        self._log(
            f"初始化完成：模式={self._current_mode_locked()}，赛季={season.season}，"
            f"发现 Pluto 设备 {len(self._discovered_devices)} 台"
        )
        capacity = self._capacity_snapshot()
        if not capacity["ok"]:
            # 业务帧变长后可能塞不进 100 ms 空口周期，必须显式暴露而不是静默跑偏
            self._log(f"空口容量告警：{capacity['note']}")

    # ==================== 内部工具 ====================

    def _log(self, msg: str) -> None:
        self._logs.appendleft(f"[{time.strftime('%H:%M:%S')}] {msg}")

    def _read_devices(self) -> List[DiscoveredDevice]:
        """执行设备扫描；扫描异常按无设备处理，不能阻断仿真启动。"""
        try:
            return list(self._device_scanner())
        except Exception:
            return []

    def _current_mode_locked(self) -> str:
        """只要任一波源绑定了真机就按真机模式展示，未绑定的波源单独标注虚拟。"""
        return "pluto" if any(self._assignments.values()) else "sim"

    def _model_for_uri_locked(self, uri: str) -> str:
        for dev in self._discovered_devices:
            if dev.uri == uri:
                return dev.model
        return "PlutoSDR"

    def _devices_snapshot_locked(self) -> List[dict]:
        role_by_uri = {uri: wave for wave, uri in self._assignments.items() if uri}
        rows: List[dict] = []
        seen = set()
        for dev in self._discovered_devices:
            rows.append({
                "uri": dev.uri,
                "model": dev.model,
                "serial": dev.serial,
                "description": dev.description,
                "role": role_by_uri.get(dev.uri),
                "source": "scan",
            })
            seen.add(dev.uri)
        # 配置文件绑定但本次扫描未出现的设备仍然展示，避免"已绑定却看不到"
        for wave, uri in self._assignments.items():
            if uri and uri not in seen:
                rows.append({
                    "uri": uri,
                    "model": "PlutoSDR",
                    "serial": "",
                    "description": "config/devices.yaml",
                    "role": wave,
                    "source": "config",
                })
        return rows

    def _role_snapshot_locked(self, wave: str) -> dict:
        # 用 get：这是每 0.5 秒推一次的快照路径，一旦因缺 key 抛异常，
        # WebSocket 会直接断开、页面失去所有状态更新，代价远大于这里少一层校验。
        uri = self._assignments.get(wave)
        if not uri:
            return {"uri": None, "model": "虚拟 SDR", "virtual": True}
        return {"uri": uri, "model": self._model_for_uri_locked(uri), "virtual": False}

    def _make_backend(self, wave: str) -> TxBackend:
        uri = self._assignments.get(wave)
        if uri:
            cycle_samples = self.season.modulation.SAMPLE_RATE_HZ // self.season.cycles_per_sec
            return PlutoTx(
                uri,
                underflow_gap_ms=self.devices.underflow_gap_ms,
                block_samples=cycle_samples,
            )

        def on_iq(iq: np.ndarray, wave: str = wave) -> None:
            # 回环解调放到独立线程：单周期解码约 15ms，若留在发射线程会挤占节拍，
            # 既让仿真跑得比实时慢，也把自身开销计入供样余量，出现虚假的负余量。
            self._decode_pool.submit(self._decode_loopback, wave, iq)

        return VirtualSdr(
            sample_rate=self.season.modulation.SAMPLE_RATE_HZ,
            on_iq=on_iq,
            underflow_gap_ms=self.devices.underflow_gap_ms,
            realtime=True,
        )

    def _decode_loopback(self, wave: str, iq: np.ndarray) -> None:
        """仿真回环自检：解码发射波形并缓存结果供诊断面板显示。"""
        try:
            result = self._loopback.decode(iq, is_info=(wave == "info"))
            self._last_loopback[wave] = {
                "ok": True,
                "air_payloads": len(result.air_payloads),
                "frames": len(result.frames),
                "cmd_ids": [hex(f.cmd_id) for f in result.frames],
            }
        except Exception as exc:  # 回环失败要暴露到界面而不是静默
            self._last_loopback[wave] = {"ok": False, "error": str(exc)}

    def _modulator(self, wave: str) -> GfskModulator:
        key = (self.team, wave)
        if key not in self._mods:
            profile = self.season.profile(self.team, wave if wave == "info" else f"jammer_{self.jammer_level}")
            self._mods[key] = GfskModulator(self.season.modulation, profile)
        return self._mods[key]

    def _rf_params(self, wave: str) -> Tuple[int, int, int]:
        """当前 (freq, bandwidth, rule_power) 三元组。"""
        name = "info" if wave == "info" else f"jammer_{self.jammer_level}"
        p = self.season.profile(self.team, name)
        return p.freq_hz, p.bandwidth_hz, p.power_dbm

    def _gen_payload(self, cmd_id: int) -> bytes:
        """生成语义合法的业务载荷（自动变化/随机按钮共用）。"""
        if cmd_id == CMD_ROBOT_POSITION:
            # 随机游走，模拟场地内移动
            for i in range(0, 12, 2):
                self._coords[i] = max(0, min(_FIELD_X_CM, self._coords[i] + random.randint(-300, 300)))
                self._coords[i + 1] = max(0, min(_FIELD_Y_CM, self._coords[i + 1] + random.randint(-200, 200)))
            self._zero_reserved_position()
            return struct.pack("<12h", *self._coords)
        if cmd_id == CMD_ROBOT_HP:
            # 表 1-42：重装/保留/3步/4步/保留/哨兵。工程与第 5 项均为保留位，必须为 0。
            hps = [random.randint(0, 600) for _ in range(6)]
            hps[1] = 0
            hps[4] = 0
            return struct.pack("<6H", *hps)
        if cmd_id == CMD_ROBOT_AMMO:
            # 表 1-43：重装/3步/4步/保留/哨兵。
            ammos = [random.randint(0, 500) for _ in range(5)]
            ammos[3] = 0
            return struct.pack("<5H", *ammos)
        if cmd_id == CMD_FIELD_STATUS:
            # 表 1-44：两个 4 字节状态字。增益点是多值枚举，且"对方未控制对应阵地"
            # 有独立编码（前哨/堡垒/基地为 4，中央高地为 3）；用 0~7 的随机数
            # 会造出协议里不存在的占位值，接收端按位域解析会得到非法状态。
            gauge = (
                random.randint(0, 1)                    # bit0 对方补给区激活
                | random.choice((0, 1, 2, 3, 4)) << 1   # bit1-3 R 堡垒增益点
                | random.choice((0, 1, 2, 3, 4)) << 4   # bit4-6 B 堡垒增益点
                | random.choice((0, 1, 2, 3, 4)) << 7   # bit7-9 R 基地增益点
                | random.choice((0, 1, 2, 3, 4)) << 10  # bit10-12 B 基地增益点
                | random.randint(0, 1) << 13            # bit13 对方制高点
                | random.choice((0, 1, 2, 3)) << 14     # bit14-15 R 中央高地
                | random.choice((0, 1, 2, 3)) << 16     # bit16-17 B 中央高地
                | random.choice((0, 1, 2, 3, 4)) << 18  # bit18-20 R 前哨
                | random.choice((0, 1, 2, 3, 4)) << 21  # bit21-23 B 前哨
            )
            terrain = (
                random.randint(0, (1 << 22) - 1)        # bit0-21 地形跨越检测点（逐位 0/1）
                | random.randint(0, 2) << 22            # bit22-23 R 前哨阵地控制
                | random.randint(0, 2) << 24            # bit24-25 B 前哨阵地控制
                | random.randint(0, 2) << 26            # bit26-27 R 基地阵地控制
                | random.randint(0, 2) << 28            # bit28-29 B 基地阵地控制
            )
            return struct.pack("<HHII", random.randint(0, 2000), 2000, gauge, terrain)
        if cmd_id == CMD_ROBOT_BUFF:
            # 表 1-45：5 个 7B 增益槽（重装/保留/3步/4步/哨兵）+ 6B 状态 + 6B 加固圆，共 47B。
            buf = bytearray()
            for slot in range(5):
                if slot == 1:  # 工程位保留，不能填随机增益
                    buf += bytes(7)
                    continue
                buf += struct.pack("<BHBBH", random.randint(0, 100), random.randint(0, 50),
                                   random.randint(0, 100), random.randint(0, 100), random.randint(0, 100))
            # 状态区 6B 全是枚举：哨兵姿态 1~6，主要状态 0~3，第 2 项保留。
            buf.append(random.randint(1, 6))                      # 35 哨兵当前姿态
            buf.append(random.randint(0, 3))                      # 36 重装主要状态
            buf.append(0)                                         # 37 保留位
            buf += bytes(random.randint(0, 3) for _ in range(3))  # 38-40 3步/4步/哨兵状态
            # 41-46 自定义装甲加固圆（x, y, r 单位 mm），半径 0 表示暂无加固情报。
            # 多数帧保持 0，偶尔给出一个合法圆面用于接收端联调。
            if random.random() < 0.3:
                buf += struct.pack("<HHH", random.randint(0, 300), random.randint(0, 200),
                                   random.randint(10, 80))
            else:
                buf += struct.pack("<HHH", 0, 0, 0)
            return bytes(buf)
        if cmd_id == CMD_INTERFERENCE_KEY:
            return self._random_key()
        raise ValueError(f"未知命令 {hex(cmd_id)}")

    def _zero_reserved_position(self) -> None:
        """表 1-41：第 2、5 组坐标（工程/空中位）为保留位，随机游走与初始化后都要清零。"""
        for idx in (2, 3, 8, 9):
            self._coords[idx] = 0

    def _random_key(self) -> bytes:
        """规则：6 位数字字母组合密钥。"""
        return "".join(random.choice(_KEY_ALPHABET)
                       for _ in range(self.season.key_len)).encode("ascii")

    # ==================== 发射线程 ====================

    def _engine_loop(self, wave: str, run_id: int) -> None:
        period = 1.0 / self.season.cycles_per_sec
        builder = SerialFrameBuilder()
        # 空口流按 15 字节连续切片且跨周期保留余量（规则附录 1：数据流连续切分）
        chunker = AirStreamChunker(
            self.season.access_code_info if wave == "info" else self.season.access_code_interfere)
        cycle_index = 0
        next_tick = time.monotonic()

        def build(idx: int) -> np.ndarray:
            """生成第 idx 周期的 IQ。

            单周期调制约 9 ms，若与推流串行，周期会变成 109 ms，10 Hz 掉到 9.2 Hz、
            字节速率只剩标称的 91%；长跑时接收端按 10 Hz 处理就会一直"慢半拍"。
            所以把它交给单独线程，与上一周期的推流重叠。
            """
            if wave == "info":
                stream = self._build_info_stream(builder)
            else:
                stream = self._build_jammer_stream(builder, idx)
            iq = modulate_air_packets(chunker.feed(stream), self._modulator(wave))
            # 不在这里补零到整周期：补零会把 9 包周期（91368 采样）抬到 100000，
            # 而每 3 个周期一次的 10 包周期本身就是 101520，平均产出 100507
            # 大于设备消耗的 100000——实测 300 周期积压 152 ms 且持续增长，
            # 发出去的内容会越来越滞后。留原始长度，块尾补零由后端统一做。
            # 调制器状态故意不复位：复位会让下一周期首符号从半频偏起步，
            # 而接收端做的是跨周期连续的符号定时。
            return iq

        # 单工作线程：调制状态（builder.seq、chunker 余量、调制器相位）只在它里面推进，
        # 主线程不并发构建，避免 seq 乱序和相位错位。
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"mod-{wave}")
        pending = None
        try:
            while not self._stop.is_set():
                with self._lock:
                    # run_id 失效说明已被 stop/MUTE 或新一轮 start 取代，本线程必须退出，
                    # 不能只看 running：新一轮 start 会把 running 重新置位，导致双线程并发推流
                    if self._run_ids[wave] != run_id:
                        break
                    running = self.info_running if wave == "info" else self.jammer_running
                if not running:
                    break
                next_tick += period  # 绝对截止时间调度，避免节拍漂移累积
                try:
                    iq = build(cycle_index) if pending is None else pending.result()
                    pending = pool.submit(build, cycle_index + 1)
                    backend = self.backends[wave]
                    # 衰减下发必须与 MUTE 的"衰减拉满"在同一把锁内，且下发前重新校验 run_id：
                    # 否则 MUTE 递增 run_id 后，已在周期体中的这次发射仍会用它持有的配置衰减
                    # （self.attenuation，如 40dB）覆盖掉刚拉满的 89dB，MUTE 硬静音偶发失效。
                    # 推流（transmit）可能阻塞（真机或仿真占位），故放在锁外，仅锁住参数下发。
                    with self._lock:
                        if self._run_ids[wave] != run_id:
                            break
                        if not (self.info_running if wave == "info" else self.jammer_running):
                            break
                        freq, bw, _ = self._rf_params(wave)
                        backend.configure(freq, bw, self.attenuation[wave],
                                          self.season.modulation.SAMPLE_RATE_HZ)
                    backend.transmit(iq)
                    cycle_index += 1
                except Exception as exc:
                    self._log(f"{wave} 发射异常：{exc}")
                    with self._lock:
                        self._run_ids[wave] += 1  # 让本线程彻底退出，避免异常后残留
                        if wave == "info":
                            self.info_running = False
                        else:
                            self.jammer_running = False
                    break
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
            time.sleep(max(0.0, next_tick - time.monotonic()))
        self._log(f"{wave} 发射已停止")

    def _build_info_stream(self, builder: SerialFrameBuilder) -> bytes:
        """信息波周期流：150B，全部为 0x0A01~0x0A05 有效帧（各命令 10Hz）。

        关闭的命令位用等长随机字节占位：保持字节速率与官方一致，
        同时保留"关闭包位填随机字节"的接收压测能力。
        """
        with self._lock:
            stream = b""
            for cmd in INFO_CMDS:
                st = self.cmds[cmd]
                if not st.enabled:
                    # 帧长 = 载荷 + 帧头5 + cmd2 + CRC16 2
                    stream += os.urandom(CMD_PAYLOAD_LEN[cmd] + 9)
                    continue
                if self.auto_vary and st.auto_vary:
                    st.payload_hex = self._gen_payload(cmd).hex()
                elif len(st.payload_hex) != CMD_PAYLOAD_LEN[cmd] * 2:
                    # 从未生成过载荷时补齐（例如开启发射前就关掉了自动变化），
                    # 否则会发出 data_length=0 的畸形帧，字节速率也不符规则
                    st.payload_hex = self._gen_payload(cmd).hex()
                stream += builder.pack(cmd, bytes.fromhex(st.payload_hex))
        return stream

    def _build_jammer_stream(self, builder: SerialFrameBuilder, cycle_index: int) -> bytes:
        """干扰波周期流：135B（规则 1350B/s = 9 空口包），1 帧 0x0A06 + 随机填充。

        密钥帧位置逐周期变化，覆盖不同 15 字节切片对齐情况。
        """
        with self._lock:
            key = self.key
        key_frame = builder.pack_key(key, self.season.key_pad_char)  # 恰好 15B = 1 个空口包载荷
        total = self.season.jammer_bytes_per_cycle
        offset = (cycle_index * 7) % AIR_PAYLOAD_LEN
        return os.urandom(offset) + key_frame + os.urandom(total - offset - len(key_frame))

    # ==================== 控制接口 ====================

    def start(self, wave: str) -> None:
        if wave not in ("info", "jammer"):
            raise ValueError("wave 必须为 info / jammer")
        with self._lock:
            if not self.safety_confirmed:
                raise ValueError("未勾选射频安全确认，禁止发射")
            if self.info_running if wave == "info" else self.jammer_running:
                return
        # 上一轮线程可能还在收尾（最多一个周期）：先等它退出再起新线程。
        # 否则"停发后立刻再开"会有新旧两个线程同时向同一台设备推流。
        old = self._threads.get(wave)
        if old is not None and old.is_alive():
            old.join(timeout=1.0)
        with self._lock:
            if wave == "info":
                self.info_running = True
            else:
                self.jammer_running = True
            # 换发新的 run_id，令任何仍在收尾的旧线程失效
            self._run_ids[wave] += 1
            run_id = self._run_ids[wave]
            # 新一轮从零相位、零滤波历史起步：中断期间射频是静的，
            # 若接着上一轮残留的相位发，开头会凭空多出一个相位跳变。
            self._modulator(wave).reset()

        # 预热设备：首次 configure 要打开 USB 上下文并重算采样时钟，实测 3~4 秒。
        # 留在发射线程里做会一直占着服务锁，WebSocket 推送和所有 API 一起卡住，
        # 用户点完"开始发射"看到的是一片无响应。放在这里只阻塞这一次请求，
        # 之后线程里的 configure 因为参数已生效会直接跳过，不重复下发。
        if self._assignments.get(wave):
            freq, bw, _ = self._rf_params(wave)
            try:
                self.backends[wave].configure(
                    freq, bw, self.attenuation[wave],
                    self.season.modulation.SAMPLE_RATE_HZ)
            except Exception as exc:
                with self._lock:
                    self._run_ids[wave] += 1
                    if wave == "info":
                        self.info_running = False
                    else:
                        self.jammer_running = False
                self._log(f"{wave} 设备初始化失败：{exc}")
                raise ValueError(f"设备初始化失败：{exc}") from exc

        thread = threading.Thread(target=self._engine_loop, args=(wave, run_id),
                                  daemon=True, name=f"tx-{wave}")
        self._threads[wave] = thread
        thread.start()
        self._log(f"{wave} 开始发射")

    def stop(self, wave: str) -> None:
        with self._lock:
            # 递增 run_id 即让当前发射线程失效：即使紧接着又点了"开始发射"，
            # 旧线程也会退出，不会与新线程同时推流
            self._run_ids[wave] += 1
            if wave == "info":
                self.info_running = False
            else:
                self.jammer_running = False
        # 立即记日志：发射线程需完成当前周期（约 100 ms）才退出，
        # 界面在等到下一次状态推送前先给出"指令已下发"的反馈
        self._log(f"{wave} 停止指令已下发")

    def mute_all(self) -> None:
        """硬静音：停发 + 衰减拉满 + 撤销安全确认。"""
        with self._lock:
            self.info_running = False
            self.jammer_running = False
            self.safety_confirmed = False
            for wave in ("info", "jammer"):
                self._run_ids[wave] += 1  # 与 stop 同理，彻底作废在跑的发射线程
            # 衰减拉满与发射周期的参数下发互斥：否则在途周期的 configure 会用配置衰减
            # 覆盖这里拉满的 89dB（MUTE 硬静音偶发失效）。backend.mute() 内部锁为
            # 服务锁→后端锁，与 configure 一致，无锁序反转。
            for backend in self.backends.values():
                backend.mute()
        self._log("全局静音（MUTE）已执行，安全确认已撤销")

    def set_safety(self, confirmed: bool) -> None:
        with self._lock:
            self.safety_confirmed = bool(confirmed)
        self._log(f"射频安全确认：{'已勾选' if confirmed else '已取消'}")

    def set_team(self, team: str) -> None:
        if team not in self.season.profiles:
            raise ValueError("阵营必须为 red / blue")
        with self._lock:
            self.team = team
            self._mods.clear()  # 阵营切换即换频点/带宽，调制器缓存失效
        self._log(f"阵营切换为 {team}")

    def set_auto_vary(self, enabled: bool) -> None:
        with self._lock:
            self.auto_vary = bool(enabled)
            if enabled:
                # 重新开启自动变化即"解锁"全部命令：否则之前手动编辑过的命令
                # 会一直停留在锁定态，看起来像自动变化开关失效
                for st in self.cmds.values():
                    st.auto_vary = True

    def set_command(self, cmd_id: int, enabled: Optional[bool] = None,
                    payload_hex: Optional[str] = None) -> None:
        self._require_info_cmd(cmd_id)
        with self._lock:
            st = self.cmds[cmd_id]
            if enabled is not None:
                st.enabled = bool(enabled)
            if payload_hex is not None:
                raw = bytes.fromhex(payload_hex)  # 非法 hex 抛 ValueError，由 API 层转 400
                expected = CMD_PAYLOAD_LEN[cmd_id]
                if len(raw) != expected:
                    raise ValueError(f"{hex(cmd_id)} 载荷必须为 {expected} 字节")
                st.payload_hex = raw.hex()
                st.auto_vary = False  # 手动编辑后锁定该命令，否则会被自动变化覆盖

    def randomize_command(self, cmd_id: int) -> None:
        self._require_info_cmd(cmd_id)
        with self._lock:
            self.cmds[cmd_id].payload_hex = self._gen_payload(cmd_id).hex()

    @staticmethod
    def _require_info_cmd(cmd_id: int) -> None:
        """仅 0x0A01~0x0A05 为信息波业务命令；越界必须报错，
        否则 self.cmds[cmd_id] 会抛 KeyError 变成 HTTP 500。"""
        if cmd_id not in INFO_CMDS:
            raise ValueError(f"不支持的命令号 {hex(cmd_id)}，仅支持 0x0A01~0x0A05")

    def set_jammer(self, level: Optional[int] = None, key: Optional[str] = None) -> None:
        # 仅切换等级时密钥留空，等价于"未提供密钥"：按规则重置为随机值，
        # 不能把空串当密钥校验，否则等级切换会被"密钥必须为数字字母组合"误拦。
        if isinstance(key, str) and not key.strip():
            key = None
        with self._lock:
            if level is not None:
                if level not in (1, 2, 3):
                    raise ValueError("干扰等级必须为 1 / 2 / 3")
                self.jammer_level = int(level)
                self._mods.pop((self.team, "jammer"), None)  # 等级切换换频点/带宽
                if key is None:
                    # 规则：干扰波难度变化时密钥重置为随机值
                    self.key = self._random_key().decode("ascii")
            if key is not None:
                raw = key.encode("ascii", errors="ignore").decode("ascii")
                if len(raw) > self.season.key_len:
                    raise ValueError(f"密钥最长 {self.season.key_len} 个 ASCII 字符")
                if not raw.isalnum():
                    raise ValueError("密钥必须为数字字母组合")
                # 补位必须落在这里而不是帧层：否则界面显示用户输入的短串、
                # 空口发出的却是补过位的 6 字节，两边对不上（规则要求恰好 6 位）。
                self.key = raw.ljust(self.season.key_len, self.season.key_pad_char)
        self._log(f"干扰参数：等级 {self.jammer_level}，密钥 {self.key}")

    def set_attenuation(self, wave: str, attenuation_db: float) -> None:
        limits = self.season.tx_limits
        if not (limits["attenuation_db_min"] <= attenuation_db <= limits["attenuation_db_max"]):
            raise ValueError("衰减超出允许范围")
        with self._lock:
            self.attenuation[wave] = float(attenuation_db)

    def device_note(self) -> str:
        """扫描为空时给出可读原因。

        "扫描不到设备"有两种完全不同的成因：缺 libiio 原生库（软件问题）
        与板子没枚举上来（连接问题）。界面必须能区分，否则只会显示
        "未发现 Pluto 设备"，让人误以为板子坏了。
        """
        if self._discovered_devices:
            return ""
        # "扫不到设备"有两种成因：缺原生库（软件）与地址不可达（网络/mDNS）。
        # 界面必须能区分，否则只会显示一句"未发现设备"，让人以为是板子坏了。
        return last_scan_note() or libiio_unavailable_reason()

    def scan_devices(self) -> List[dict]:
        """重新扫描 PlutoSDR，返回设备列表（含当前角色绑定）。"""
        devices = self._read_devices()
        with self._lock:
            self._discovered_devices = devices
            rows = self._devices_snapshot_locked()
        self._log(f"设备扫描完成：发现 {len(devices)} 台 PlutoSDR")
        return rows

    def set_device_role(self, uri: str, role: Optional[str]) -> None:
        """把扫描到的设备分配给信息波/干扰波，或取消分配。

        同一波源只允许绑定一台设备；切换会重建对应后端。发射线程仍在运行时
        禁止换绑，否则会关闭正在推流的设备，导致线程访问已释放的 SDR。
        """
        role = role or None
        if role not in (None, "info", "jammer"):
            raise ValueError("设备角色必须为 info / jammer / 空")
        with self._lock:
            assigned_by_uri = {u: w for w, u in self._assignments.items() if u}
            known = {dev.uri for dev in self._discovered_devices}
            # 允许操作配置文件绑定但本次扫描未出现的设备（例如临时离线）
            if uri not in known and uri not in assigned_by_uri:
                raise ValueError("设备不在扫描列表中，请重新扫描")
            current = assigned_by_uri.get(uri)
            if current == role:
                return
            affected = {wave for wave in (current, role) if wave}
            for wave in affected:
                running = self.info_running if wave == "info" else self.jammer_running
                thread = self._threads.get(wave)
                if running or (thread is not None and thread.is_alive()):
                    name = "信息波" if wave == "info" else "干扰波"
                    raise ValueError(f"{name}正在发射或收尾，请先停止后再切换设备")

            if role:
                old_uri = self._assignments[role]
                if old_uri and old_uri != uri:
                    self._assignments[role] = None
            if current:
                self._assignments[current] = None
            if role:
                self._assignments[role] = uri

            # 受影响波源统一重建后端：解除绑定的回到虚拟 SDR，绑定真机的切到 Pluto
            for wave in affected:
                self.backends[wave].close()
                self.backends[wave] = self._make_backend(wave)

            model = self._model_for_uri_locked(uri)
            if role is None:
                self._log(f"设备 {model}（{uri}）已取消分配")
            else:
                name = "信息波" if role == "info" else "干扰波"
                self._log(f"设备 {model}（{uri}）已分配为{name}发射")

    def shutdown(self) -> None:
        self._stop.set()
        with self._lock:
            self.info_running = False
            self.jammer_running = False
        for t in self._threads.values():
            t.join(timeout=2.0)
        self._decode_pool.shutdown(wait=False, cancel_futures=True)
        for backend in self.backends.values():
            backend.close()
        self._log("服务已关闭")

    # ==================== 状态快照（Web） ====================

    def snapshot(self) -> dict:
        with self._lock:
            commands = [
                {
                    "cmd_id": hex(cmd),
                    "name": CMD_NAMES[cmd],
                    "enabled": self.cmds[cmd].enabled,
                    "payload_hex": self.cmds[cmd].payload_hex,
                    # 全局开关与命令自身状态取"与"：全局关闭即全部冻结在该命令当前载荷，
                    # 命令被手动编辑后也应锁定（取"或"会让全局开关失效并覆盖手动载荷）
                    "auto_vary": self.auto_vary and self.cmds[cmd].auto_vary,
                }
                for cmd in INFO_CMDS
            ]
            info_freq, info_bw, info_power = self._rf_params("info")
            jam_freq, jam_bw, jam_power = self._rf_params("jammer")
            return {
                "season": self.season.season,
                "mode": self._current_mode_locked(),
                "devices": self._devices_snapshot_locked(),
                "device_note": self.device_note(),
                "device_roles": {wave: self._role_snapshot_locked(wave) for wave in ("info", "jammer")},
                "rules": self._rules_snapshot(),
                "team": self.team,
                "safety_confirmed": self.safety_confirmed,
                "auto_vary": self.auto_vary,
                "key": self.key,
                "commands": commands,
                "info": {
                    "running": self.info_running,
                    "freq_hz": info_freq, "bandwidth_hz": info_bw, "power_dbm": info_power,
                    "attenuation_db": self.attenuation["info"],
                    "stats": self.backends["info"].stats.__dict__.copy(),
                    "loopback": self._last_loopback.get("info", {}),
                },
                "jammer": {
                    "running": self.jammer_running,
                    "level": self.jammer_level,
                    "freq_hz": jam_freq, "bandwidth_hz": jam_bw, "power_dbm": jam_power,
                    "attenuation_db": self.attenuation["jammer"],
                    "stats": self.backends["jammer"].stats.__dict__.copy(),
                    "loopback": self._last_loopback.get("jammer", {}),
                },
                "logs": list(self._logs),
            }

    def _capacity_snapshot(self) -> dict:
        """核算信息波业务字节能否塞进一个空口周期。

        2027 协议把 0x0A04/0x0A05 加长后，5 条命令合计 150B/周期；空口包 27B
        只装 15B 载荷，等效放大 1.8 倍。2027 规则手册尚未给出新的比特率/空口格式，
        这里只做核算并把差距显式暴露，不改动射频参数。
        """
        mod = self.season.modulation
        payload_bytes = self.season.info_bytes_per_cycle
        air_bytes = payload_bytes * AIR_FRAME_LEN / AIR_PAYLOAD_LEN
        air_bits = air_bytes * 8
        air_seconds = air_bits / mod.bit_rate_hz
        budget_s = 1.0 / self.season.cycles_per_sec
        required_bps = air_bits / budget_s
        ok = air_seconds <= budget_s + 1e-9
        note = ""
        if not ok:
            note = (f"信息波 {payload_bytes}B/{budget_s * 1000:.0f}ms 需 {required_bps:.0f} bps，"
                    f"当前码率 {mod.bit_rate_hz:.1f} bps（{air_seconds * 1000:.2f}ms/周期），"
                    f"实际将以 {budget_s / air_seconds * 100:.1f}% 实时发送；"
                    f"待 2027 规则手册确认空口/比特率参数")
        return {
            "payload_bytes": payload_bytes,
            "air_bytes": round(air_bytes, 1),
            "air_ms": round(air_seconds * 1000, 2),
            "budget_ms": round(budget_s * 1000, 1),
            "required_bps": round(required_bps),
            "bit_rate_bps": round(mod.bit_rate_hz, 1),
            "ok": ok,
            "note": note,
        }

    def _rules_snapshot(self) -> dict:
        """规则核对数据：表 5-23 全部波源参数 + 调制参数，供 Web 面板显示。"""
        mod = self.season.modulation
        profiles = []
        for team in ("red", "blue"):
            for wave in ("info", "jammer_1", "jammer_2", "jammer_3"):
                p = self.season.profile(team, wave)
                profiles.append({
                    "label": WAVE_LABELS[(team, wave)],
                    "team": team,
                    "wave": wave,
                    "freq_hz": p.freq_hz,
                    "bandwidth_hz": p.bandwidth_hz,
                    "power_dbm": p.power_dbm,
                    # 调频灵敏度 = 2π·Δf/SampleRate（rad/sample），与规则表 5-23 一致
                    "sensitivity": round(2 * math.pi * p.freq_deviation_hz / mod.SAMPLE_RATE_HZ, 4),
                })
        return {
            "source": "雷达业务帧：通信协议 V1.0.0 §1.5 · 射频/空口：规则手册 V2.2.0 表 5-23/附录 1",
            "modulation": "2-GFSK · SPS=47 · BT=0.35 · 采样率 1MS/s · 无前导码/白化/FEC",
            "access_code_info": self.season.access_code_info.hex().upper(),
            "access_code_interfere": self.season.access_code_interfere.hex().upper(),
            "info_rate": "1500 B/s · 0x0A01~0x0A05 全有效数据 · 各命令 10Hz（150B/周期）",
            "jammer_rate": "1350 B/s · 0x0A06 10Hz + 随机填充",
            "capacity": self._capacity_snapshot(),
            "profiles": profiles,
        }
