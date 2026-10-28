# -*- coding: utf-8 -*-
"""发射后端掉线自愈的回归测试。

Pluto 走 USB 时掉线重连会换端口号（实测 usb:2.13.5 → usb:2.15.5 → usb:2.6.5），
只记住原始 URI 就再也定位不到板子。这里固化两件事：
一是只有"掉线"类异常才触发重连，配置错误必须原样上抛；
二是重连后要把上一次的射频参数补回去，否则会用默认频点继续发。
"""

import sys
import types

import pytest

from radar_tx.device.pluto_tx import PlutoTx, _is_disconnect, resolve_uri


class FakeSdr:
    """记录属性写入的最小 Pluto 替身。"""

    def __init__(self) -> None:
        self.sample_rate = 0
        self.tx_lo = 0
        self.tx_rf_bandwidth = 0
        self.tx_hardwaregain_chan0 = 0
        self.tx_cyclic_buffer = True
        self.serial = "fake-serial-0001"


@pytest.fixture()
def fake_hardware(monkeypatch):
    """让 _ensure_device 在没有真实设备的情况下也能走通。"""
    monkeypatch.setattr("radar_tx.device.libiio_runtime.ensure_libiio_available",
                        lambda: (True, ""))
    # 扫描也要挡掉：真机扫描一次要几百毫秒，测试不该依赖硬件在场
    monkeypatch.setattr("radar_tx.device.discovery.scan_pluto_devices", lambda: [])
    created = []

    def _make(_uri):
        sdr = FakeSdr()
        created.append(sdr)
        return sdr

    fake_adi = types.ModuleType("adi")
    fake_adi.Pluto = _make
    monkeypatch.setitem(sys.modules, "adi", fake_adi)
    return created


def test_disconnect_detected_for_usb_errors():
    assert _is_disconnect(OSError(110, "host unreachable"))
    assert _is_disconnect(Exception("No device found"))
    assert _is_disconnect(OSError(19, "No such device"))


def test_config_errors_are_not_treated_as_disconnect():
    """缓冲长度变更、参数非法属于配置问题，重连只会掩盖真正的错误。"""
    assert not _is_disconnect(ValueError("Cannot change buffer length on the fly"))
    assert not _is_disconnect(RuntimeError("衰减超出允许范围"))


def test_apply_config_is_remembered(fake_hardware):
    tx = PlutoTx("usb:fake")
    tx._serial = "fake-serial-0001"  # 预设 serial，避免触发真实设备扫描
    tx._apply_config(433_200_000, 250_000, 40.0, 1_000_000)
    assert tx._last_config == (433_200_000, 250_000, 40.0, 1_000_000)


def test_reconnect_replays_rf_config(fake_hardware):
    """掉线重连后必须按上一次的配置恢复频点、带宽与增益。"""
    tx = PlutoTx("usb:fake")
    tx._serial = "fake-serial-0001"  # 预设 serial，避免触发真实设备扫描
    tx._last_config = (433_200_000, 250_000, 40.0, 1_000_000)
    tx._sdr = None
    tx._ensure_device()
    sdr = fake_hardware[0]
    assert sdr.tx_lo == 433_200_000
    # 窄带波形要留模拟滤波余量，重放时同样走这条策略
    assert sdr.tx_rf_bandwidth == 540_000
    assert sdr.tx_hardwaregain_chan0 == -40.0
    assert sdr.sample_rate == 1_000_000


def test_reconnect_records_reason(fake_hardware):
    tx = PlutoTx("usb:fake")
    tx._reconnect(OSError(110, "host unreachable"))
    assert tx.stats.extra["reconnects"] == 1
    assert "host unreachable" in tx.stats.extra["last_error"]
    assert tx.stats.connected is True


def test_plain_uri_passes_through():
    assert resolve_uri("usb:2.6.5") == "usb:2.6.5"
    assert resolve_uri("ip:192.168.2.1") == "ip:192.168.2.1"


def test_serial_uri_resolves_to_current_port(monkeypatch):
    """按 serial 绑定：端口号变了也能找到同一块板。"""
    from radar_tx.device import discovery

    class Dev:
        def __init__(self, uri, serial):
            self.uri, self.serial = uri, serial

    monkeypatch.setattr(discovery, "scan_pluto_devices",
                        lambda: [Dev("usb:9.9.9", "03df6d62f0af"), Dev("usb:8.8.8", "03df62bac557")])
    assert resolve_uri("serial:03df6d62") == "usb:9.9.9"
    assert resolve_uri("serial:03df62ba") == "usb:8.8.8"
    with pytest.raises(RuntimeError):
        resolve_uri("serial:deadbeef")
