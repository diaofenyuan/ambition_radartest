# -*- coding: utf-8 -*-
"""雷达无线电自测模拟发射系统入口。

用法：python app.py [--host 127.0.0.1] [--port 8780] [--mode sim|pluto]
"""

from __future__ import annotations

import argparse

import uvicorn

from radar_tx.config import load_devices_config, load_season_config
from radar_tx.scheduler.tx_service import TxService
from web.api import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="雷达无线电自测模拟发射系统")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8780)
    parser.add_argument("--mode", choices=["sim", "pluto"], default=None,
                        help="覆盖 config/devices.yaml 中的运行模式")
    args = parser.parse_args()

    season = load_season_config()
    devices = load_devices_config()
    if args.mode:
        devices.mode = args.mode

    service = TxService(season, devices)
    def request_exit() -> None:
        server.should_exit = True

    app = create_app(service, on_ui_closed=request_exit)
    server = uvicorn.Server(uvicorn.Config(
        app, host=args.host, port=args.port, log_level="warning",
    ))
    print(f"发射系统启动：http://{args.host}:{args.port}  模式={devices.mode}")
    server.run()


if __name__ == "__main__":
    main()
