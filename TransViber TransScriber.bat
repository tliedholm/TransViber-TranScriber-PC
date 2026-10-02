@echo off
rem Launches the app without a console window.
if not exist "%~dp0.venv\Scripts\pythonw.exe" (
    echo The app's Python environment is missing from "%~dp0.venv".
    echo Restore the environment or use the standalone app.
    pause
    exit /b 1
)
echo === Launcher clicked %date% %time% ===>> "%~dp0app.log"
start "" /D "%~dp0" "%~dp0.venv\Scripts\pythonw.exe" "%~dp0app\main.py"
