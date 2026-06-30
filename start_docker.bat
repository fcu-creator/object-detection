@echo off
setlocal
cd /d "%~dp0"

set "CAMERA_HOST_IP="
for /f "usebackq tokens=*" %%I in (`powershell -NoProfile -Command "$items = [System.Net.NetworkInformation.NetworkInterface]::GetAllNetworkInterfaces() ^| Where-Object { $_.OperationalStatus -eq 'Up' -and $_.NetworkInterfaceType -ne 'Loopback' } ^| ForEach-Object { $_.GetIPProperties().UnicastAddresses } ^| Where-Object { $_.Address.AddressFamily -eq [System.Net.Sockets.AddressFamily]::InterNetwork -and -not $_.Address.ToString().StartsWith('169.254.') }; $preferred = $items ^| Where-Object { $_.Address.ToString().StartsWith('192.168.') } ^| Select-Object -First 1; if (-not $preferred) { $preferred = $items ^| Where-Object { -not $_.Address.ToString().StartsWith('127.') } ^| Select-Object -First 1 }; if ($preferred) { $preferred.Address.ToString().Trim() }"`) do if not defined CAMERA_HOST_IP set "CAMERA_HOST_IP=%%I"

if not defined CAMERA_HOST_IP set "CAMERA_HOST_IP=127.0.0.1"
echo(%CAMERA_HOST_IP%| findstr /r /x "[0-9][0-9]*\.[0-9][0-9]*\.[0-9][0-9]*\.[0-9][0-9]*" >nul
if errorlevel 1 set "CAMERA_HOST_IP=127.0.0.1"
> ".env" echo CAMERA_HOST_IP=%CAMERA_HOST_IP%

docker compose up --build -d --remove-orphans
if errorlevel 1 (
    echo.
    echo Docker startup failed. Make sure Docker Desktop is running.
    pause
    exit /b 1
)

for /l %%N in (1,1,30) do (
    if exist "certs\camera.crt" goto certificate_ready
    timeout /t 1 /nobreak >nul
)

:certificate_ready
if exist "certs\camera.crt" (
    certutil -user -addstore Root "certs\camera.crt" >nul 2>&1
)

echo.
echo Open on this computer:
echo https://localhost:8001
echo.
echo LAN address for another device:
echo https://%CAMERA_HOST_IP%:8001
echo.
if "%CAMERA_HOST_IP:~0,8%"=="192.168." (
    echo LAN IP is in the requested 192.168.x.x range.
) else (
    echo Current LAN IP is %CAMERA_HOST_IP%.
    echo Connect this PC to the teacher's 192.168.x.x router, then run this file again.
)
start "" "https://localhost:8001"
pause
