# -*- coding: utf-8 -*-
"""发射服务状态机关键链路测试。

覆盖本次修复的四类缺陷：
- 越界命令号必须报错而不是抛 KeyError（否则接口变 500）；
- 自动变化开关的"与"语义（全局开关必须真正生效，手动编辑后锁定）；
- 关闭自动变化时不得发出 data_length=0 的畸形帧；
- 停发后立刻再开不得残留旧发射线程（双线程并发推流）。
"""

import struct
import time

import numpy as np
import pytest

from radar_tx.config import load_devices_config, load_season_config
from radar_tx.device import discovery
from radar_tx.device.discovery import DiscoveredDevice, last_scan_note, scan_pluto_devices
from radar_tx.device.pluto_tx import PlutoTx, _set_tx_gain, _tx_gain_attr
from radar_tx.device.virtual_sdr import VirtualSdr
from radar_tx.protocol.serial_frame import (
    CMD_FIELD_STATUS, CMD_INTERFERENCE_KEY, CMD_ROBOT_BUFF, CMD_ROBOT_POSITION,
    SerialFrameBuilder, SerialFrameDecoder,
)
from radar_tx.scheduler.tx_service import TxService


@pytest.fixture
def service():
    # 测试不扫描真实硬件，避免本机 USB 设备影响状态断言
    svc = TxService(load_season_config(), load_devices_config(), device_scanner=lambda: [])
    yield svc
    svc.shutdown()


def test_unknown_command_id_rejected(service):
    # 0x0A06 属于干扰波密钥、0xFFFF 非法：都必须抛 ValueError 由接口层转 400
    with pytest.raises(ValueError):
        service.set_command(0x0A06, enabled=True)
    with pytest.raises(ValueError):
        service.set_command(0xFFFF, payload_hex="00")
    with pytest.raises(ValueError):
        service.randomize_command(0xFFFF)


def test_payload_length_checked_against_command(service):
    with pytest.raises(ValueError):
        service.set_command(CMD_ROBOT_POSITION, payload_hex="00" * 10)  # 0x0A01 固定 24B
    service.set_command(CMD_ROBOT_POSITION, payload_hex="00" * 24)


def test_auto_vary_master_switch_and_manual_lock(service):
    def flags():
        return {c["cmd_id"]: c["auto_vary"] for c in service.snapshot()["commands"]}

    # 开局全自动
    assert all(flags().values())
    # 全局关闭后全部冻结（此前取"或"，全局开关根本不生效）
    service.set_auto_vary(False)
    assert not any(flags().values())
    # 重新开启即解锁全部命令
    service.set_auto_vary(True)
    assert all(flags().values())
    # 手动编辑后该命令锁定，且不因全局开启被解锁
    service.set_command(CMD_ROBOT_POSITION, payload_hex="11" * 24)
    assert flags()[hex(CMD_ROBOT_POSITION)] is False
    assert sum(flags().values()) == len(flags()) - 1


def test_info_stream_keeps_140_bytes_without_auto_vary(service):
    # 开启发射前就关掉自动变化时，命令从未生成过载荷；
    # 若不补齐就会发出 data_length=0 的畸形帧，且周期字节数不再是 140
    service.set_auto_vary(False)
    stream = service._build_info_stream(SerialFrameBuilder())
    assert len(stream) == 140


def test_buff_tail_fields_are_legal_enums(service):
    # 表 1-48：尾部 6B 是枚举而不是自由数据——哨兵姿态 1~6、各机器人主要状态 0~3。
    # 早先这里填 os.urandom(6)，约 97% 的帧会带上不存在的姿态值。
    for _ in range(200):
        payload = service._gen_payload(CMD_ROBOT_BUFF)
        assert len(payload) == 41
        assert 1 <= payload[35] <= 6, f"哨兵姿态越界：{payload[35]}"
        assert all(0 <= v <= 3 for v in payload[36:]), f"状态枚举越界：{payload[36:].hex()}"


def test_field_status_flags_respect_bitfield_ranges(service):
    # 表 1-47：中央高地与前哨站都是 0/1/2 的三值枚举，随机取 0~7 会造出非法的 3
    for _ in range(200):
        flags = struct.unpack_from("<I", service._gen_payload(CMD_FIELD_STATUS), 4)[0]
        assert (flags >> 1) & 0x3 <= 2, "中央高地占位值越界"
        assert (flags >> 6) & 0x3 <= 2, "前哨站占位值越界"
        assert flags >> 16 == 0, "表 1-47 只定义到 bit15"


def test_short_key_pads_ui_and_wire_consistently(service):
    # 规则要求恰好 6 位。补位若只发生在帧层，界面显示 "AB1"、空口却发 "AB1000"，
    # 两边不一致；现在补位在状态层完成，快照即实发内容。
    service.set_jammer(key="AB1")
    assert service.snapshot()["key"] == "AB1000"
    stream = service._build_jammer_stream(SerialFrameBuilder(), 0)
    keys = [f.payload for f in SerialFrameDecoder().feed(stream)
            if f.cmd_id == CMD_INTERFERENCE_KEY]
    assert keys == [b"AB1000"]


def test_rapid_restart_leaves_no_stale_tx_thread(service):
    service.set_safety(True)
    service.start("info")
    old = service._threads["info"]
    time.sleep(0.25)
    service.stop("info")
    service.start("info")  # 停发后立刻再开
    time.sleep(0.3)
    assert not old.is_alive(), "旧发射线程未退出，会与新线程并发推流"
    assert service._threads["info"].is_alive()


def test_mute_pulls_attenuation_to_max_despite_inflight_cycle(service):
    # MUTE 的"衰减拉满"必须与在途发射周期的参数下发互斥：
    # 修复前，已在周期体中的发射会用配置衰减(40dB)覆盖刚拉满的 89dB，偶发失效。
    # 反复在发射中段触发 MUTE，验证衰减恒为 89。
    for _ in range(20):
        service.set_safety(True)  # MUTE 会撤销安全确认，每轮重新勾选
        service.start("info")
        time.sleep(0.12)  # 让发射线程进入周期体
        service.mute_all()
        snap = service.snapshot()
        assert snap["info"]["running"] is False
        assert snap["info"]["stats"]["attenuation_db"] == 89.0, "MUTE 衰减被在途周期覆盖"
    service.stop("info")


DEVICE = DiscoveredDevice(
    uri="usb:1.6.5",
    description="0456:b673 (Analog Devices Inc. PlutoSDR (ADALM-PLUTO)), serial=03df62",
    serial="03df62",
    model="PlutoSDR",
)


def test_scan_dedupes_same_board_and_ignores_non_pluto(monkeypatch):
    # 装了 pylibiio 但缺 libiio 原生库时抛 TypeError 而非 ImportError，
    # importorskip 兜不住，需自行降级为 skip。
    try:
        import iio
    except Exception:
        pytest.skip("未加载到 libiio 原生库，跳过真机扫描用例")

    contexts = {
        "usb:1.6.5": "0456:b673 (Analog Devices Inc. PlutoSDR (ADALM-PLUTO)), serial=03df62",
        "ip:pluto.local": "fe80::1 (Analog Devices PlutoSDR Rev.B), serial=03df62",
        "ip:other.local": "192.168.1.9 (Some Other IIO Device), serial=other",
    }
    monkeypatch.setattr(iio, "scan_contexts", lambda: contexts)
    # mDNS 名字的连通性探测会走真实网络，测试里固定为不可达，保证确定性
    monkeypatch.setattr(discovery, "_reachable", lambda uri, timeout_s=2.0: False)
    devices = scan_pluto_devices()
    assert len(devices) == 1
    assert devices[0].uri == "usb:1.6.5"  # 同一块板优先保留 USB
    assert devices[0].serial == "03df62"


def test_scan_drops_unreachable_mdns_uri(monkeypatch):
    # ip:pluto.local 在没有 mDNS 解析器的机器上会被普通 DNS 抢答（实测解析到公网
    # 198.20.0.21），照抄它绑定设备必然连不上。既然实测不通就必须丢弃，
    # 而不是把坏地址摆在界面上让人点。
    try:
        import iio
    except Exception:
        pytest.skip("未加载到 libiio 原生库，跳过真机扫描用例")

    monkeypatch.setattr(iio, "scan_contexts",
                        lambda: {"ip:pluto.local": "fe80::1 (PlutoSDR Rev.B), serial=abc"})
    monkeypatch.setattr(discovery, "_reachable", lambda uri, timeout_s=2.0: False)
    assert scan_pluto_devices() == []
    assert "ip:pluto.local" in last_scan_note()


def test_mdns_probe_only_for_hostname_uris():
    assert discovery._needs_reachability_probe("ip:pluto.local") is True
    assert discovery._needs_reachability_probe("ip:192.168.2.1") is False
    assert discovery._needs_reachability_probe("usb:1.6.5") is False


def test_tx_gain_attr_ignores_stale_instance_attribute():
    # 历史代码写 sdr.tx_hardwaregain = -40 时不会报错，只会给对象挂一个无效属性。
    # 探测必须仍然认准硬件属性名，否则修完还会被脏属性骗一次。
    class Modern:
        tx_hardwaregain_chan0 = 0

    sdr = Modern()
    sdr.tx_hardwaregain = -40          # 模拟旧 bug 留下的脏属性
    assert _tx_gain_attr(sdr) == "tx_hardwaregain_chan0"


def test_set_tx_gain_reaches_hardware_attribute():
    class Legacy:
        tx_hardwaregain = 0

    class Modern:
        tx_hardwaregain_chan0 = 0

    modern = Modern()
    _set_tx_gain(modern, 40)
    assert modern.tx_hardwaregain_chan0 == -40
    legacy = Legacy()
    _set_tx_gain(legacy, 89)
    assert legacy.tx_hardwaregain == -89


def test_device_role_switch_rebuilds_backend():
    svc = TxService(load_season_config(), load_devices_config(), device_scanner=lambda: [DEVICE])
    try:
        with pytest.raises(ValueError):
            svc.set_device_role("usb:999", "info")  # 未扫描到的设备不允许直接绑定
        assert isinstance(svc.backends["info"], VirtualSdr)
        svc.set_device_role(DEVICE.uri, "info")
        assert isinstance(svc.backends["info"], PlutoTx)
        assert isinstance(svc.backends["jammer"], VirtualSdr)
        snap = svc.snapshot()
        assert snap["mode"] == "pluto"
        assert snap["device_roles"]["info"]["uri"] == DEVICE.uri
        assert snap["devices"][0]["role"] == "info"

        # 同一块板改绑到干扰波，信息波必须回到虚拟后端
        svc.set_device_role(DEVICE.uri, "jammer")
        assert isinstance(svc.backends["info"], VirtualSdr)
        assert isinstance(svc.backends["jammer"], PlutoTx)
        assert svc.snapshot()["devices"][0]["role"] == "jammer"

        svc.set_device_role(DEVICE.uri, None)
        assert isinstance(svc.backends["jammer"], VirtualSdr)
        assert svc.snapshot()["mode"] == "sim"
    finally:
        svc.shutdown()


def test_device_role_switch_rejected_while_running():
    svc = TxService(load_season_config(), load_devices_config(), device_scanner=lambda: [DEVICE])
    try:
        svc.set_safety(True)
        svc.start("info")
        with pytest.raises(ValueError):
            svc.set_device_role(DEVICE.uri, "info")
        svc.stop("info")
    finally:
        svc.shutdown()


def test_pluto_tx_keeps_fixed_buffer_length_across_cycles():
    # 信息波周期长度会在 9 包(91368)与 10 包(101520)之间变化；
    # 非循环 TX 缓冲长度不可变，必须跨周期保留余量并按固定块推送。
    class FakeSdr:
        def __init__(self):
            self.lengths = []

        def tx(self, data):
            self.lengths.append(len(data))

    backend = PlutoTx("usb:test", block_samples=100000)
    backend._sdr = FakeSdr()
    backend._push(np.zeros(91368, dtype=np.complex64))
    backend._push(np.zeros(101520, dtype=np.complex64))
    backend._push(np.zeros(100000, dtype=np.complex64))
    assert backend._sdr.lengths == [100000, 100000, 100000]
    assert backend._pending.size == 1520
    backend.close()

