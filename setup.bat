@echo off
setlocal

:: Define paths
set "PROJECT_DIR=%~dp0"
set "VENV_DIR=%PROJECT_DIR%venv"
set "SCRIPTS_DIR=%PROJECT_DIR%scripts"
set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"
set "SHORTCUT_PS1=%SCRIPTS_DIR%\create_shortcut.ps1"
set "RUN_BAT=%PROJECT_DIR%run.bat"

echo [1/4] Checking python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed. Please install Python 3.9 or higher.
    pause
    exit /b 1
)

echo [2/4] Creating virtual environment...
if not exist "%VENV_DIR%" (
    python -m venv "%VENV_DIR%"
)

echo [3/4] Installing dependencies...
"%PYTHON_EXE%" -m pip install --upgrade pip
"%PYTHON_EXE%" -m pip install -r requirements.txt

echo [4/4] Creating desktop shortcut...
:: Create run.bat if it doesn't exist
(
echo @echo off
echo cd /d "%%~dp0"
echo call venv\Scripts\activate
echo python run.py
echo pause
) > "%RUN_BAT%"

:: Run powershell to create shortcut
powershell -ExecutionPolicy Bypass -File "%SHORTCUT_PS1%" -TargetFile "%RUN_BAT%" -ShortcutName "AIDetector" -Description "Launch Offline AI Detector"

echo.
echo Setup Complete!
echo You can now use the 'AIDetector' shortcut on your Desktop.
pause
