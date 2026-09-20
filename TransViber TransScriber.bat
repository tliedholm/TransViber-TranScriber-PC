@echo off
rem Launches the app without a console window.
start "" "%~dp0.venv\Scripts\pythonw.exe" "%~dp0app\main.py"
