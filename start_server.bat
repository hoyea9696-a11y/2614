@echo off
title AI Vision Assistant Platform
echo ==========================================
echo   AI Vision Assistant - Starting Server
echo ==========================================
echo.

echo [1/4] Cleaning up port 8005 (if occupied)...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8005') do (
taskkill /F /PID %%a 2>nul
)

echo [2/4] Activating Virtual Environment...
cd /d %~dp0
if not exist "venv" (
    echo Error: venv directory not found!
    pause
    exit /b
)
call venv\Scripts\activate

echo [3/4] Checking environment variables...
if not exist ".env" (
    echo Warning: .env file not found!
)

echo [4/4] Starting Backend Server...
echo.
python server\main.py

echo.
echo Server has stopped.
pause
