@echo off
chcp 65001 >nul
cd /d "%~dp0"
setlocal enabledelayedexpansion

REM 雷达无线电自测模拟发射系统 · Windows 一键停止
REM start.bat 的配套脚本：先下发 MUTE 让真机落到最大衰减（直接杀进程时射频可能仍在发射），
REM 再结束发射服务进程并回收端口。--port 必须与启动时一致，未指定按默认 8780。

set "PORT=8780"
if not "%~1"=="" (
    set "PREV="
    for %%A in (%*) do (
        if defined PREV (
            if /I "!PREV!"=="--port" set "PORT=%%~A"
        )
        set "PREV=%%~A"
    )
)

echo [停止] 结束发射服务并清理后台进程 ...  端口 %PORT%
echo.
set "PORT=%PORT%"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$port=$env:PORT; $url='http://127.0.0.1:' + $port + '/api/mute'; try { Invoke-WebRequest -Uri $url -Method Post -TimeoutSec 2 -UseBasicParsing | Out-Null; Write-Host '[1/3] 已下发 MUTE：发射停止、衰减拉满、安全确认撤销' } catch { Write-Host '[1/3] 服务未响应（可能已停止），跳过 MUTE' }; $procs=Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' -and $_.CommandLine -like '*app.py*' -and ( ($_.CommandLine -like ('*--port ' + $port + '*')) -or ($port -eq '8780' -and $_.CommandLine -notlike '*--port*') ) }; if ($procs) { foreach ($p in $procs) { Write-Host ('[2/3] 结束 PID ' + $p.ProcessId); Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue } } else { Write-Host '[2/3] 未发现运行中的发射服务进程' }; Start-Sleep -Milliseconds 800; $listen=Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1; if ($listen) { Write-Host ('[3/3] 端口仍被 PID ' + $listen.OwningProcess + ' 占用，强制结束'); taskkill /T /F /PID $listen.OwningProcess | Out-Null } else { Write-Host '[3/3] 端口已释放' }"
echo.
pause
