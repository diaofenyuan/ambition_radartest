# -*- coding: utf-8 -*-
"""发射服务状态机关键链路测试。

覆盖本次修复的四类缺陷：
- 越界命令号必须报错而不是抛 KeyError（否则接口变 500）；
- 自动变化开关的"与"语义（全局开关必须真正生效，手动编辑后锁定）；
- 关闭自动变化时不得发出 data_length=0 的畸形帧；
- 停发后立刻再开不得残留旧发射线程（双线程并发推流）。
"""

import time

import pytest

from radar_tx.config import load_devices_config, load_season_config
from radar_tx.protocol.serial_frame import CMD_ROBOT_POSITION, SerialFrameBuilder
from radar_tx.scheduler.tx_service import TxService


@pytest.fixture
def service():
    svc = TxService(load_season_config(), load_devices_config())
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

