@echo off
chcp 65001 >nul
cd /d "%~dp0"
setlocal enabledelayedexpansion

REM 雷达无线电自测模拟发射系统 · Windows 一键启动
REM 首次运行自动创建 .venv 并安装依赖（清华镜像），之后直接启动服务。
REM 可透传参数，例如：start.bat --mode pluto --port 8780

set "VENV=.venv"
set "VPY=%VENV%\Scripts\python.exe"
set "MARKER=%VENV%\.deps_ok"
set "MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple"

REM 解析 --host/--port，用于启动后自动打开控制台页面（未指定则用默认值）
set "HOST=127.0.0.1"
set "PORT=8780"
if not "%~1"=="" (
    set "PREV="
    for %%A in (%*) do (
        if defined PREV (
            if /I "!PREV!"=="--host" set "HOST=%%~A"
            if /I "!PREV!"=="--port" set "PORT=%%~A"
        )
        set "PREV=%%~A"
    )
)
REM 监听 0.0.0.0 时本机访问仍走回环地址
if "%HOST%"=="0.0.0.0" set "HOST=127.0.0.1"

REM 1) 定位基础 Python 解释器
set "PY="
where python >nul 2>nul && set "PY=python"
if not defined PY where py >nul 2>nul && set "PY=py -3"
if not defined PY (
    echo [错误] 未检测到 Python，请先安装 Python 3.10+ 并加入 PATH。
    pause
    exit /b 1
)

REM 2) 首次运行：建立虚拟环境（隔离依赖，避免污染系统 Python）
if not exist "%VPY%" (
    echo [初始化] 正在创建虚拟环境 %VENV% ...
    %PY% -m venv "%VENV%"
    if errorlevel 1 (
        echo [错误] 虚拟环境创建失败。
        pause
        exit /b 1
    )
)

REM 3) 安装依赖：以标记文件保证只装一次，也避免安装中断后重复触发
if not exist "%MARKER%" (
    echo [初始化] 正在安装依赖（清华镜像）...
    "%VPY%" -m pip install --upgrade pip -i %MIRROR%
    if errorlevel 1 echo [提示] pip 升级失败，使用当前版本继续。
    "%VPY%" -m pip install -i %MIRROR% -r requirements.txt
    if errorlevel 1 (
        echo [提示] 完整依赖安装失败（通常为真机可选依赖 pyadi-iio），改用核心依赖重装 ...
        "%VPY%" -m pip install -i %MIRROR% numpy fastapi uvicorn websockets pyyaml pydantic
        if errorlevel 1 (
            echo [错误] 核心依赖安装失败，请检查网络后重试。
            pause
            exit /b 1
        )
    )
    echo done > "%MARKER%"
    echo [初始化] 依赖安装完成。
)

REM 4) 启动服务，并延时自动打开控制台页面
echo [启动] 雷达无线电自测模拟发射系统 ...  http://%HOST%:%PORT%
echo.
REM 独立后台进程等待端口就绪后打开浏览器，避免与前台服务互阻塞
start "" /b powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 2; Start-Process 'http://%HOST%:%PORT%'"
"%VPY%" app.py %*

echo.
echo 服务已退出。
pause
