@echo off
rem Start Subtitles without a console window (pythonw).
cd /d "%~dp0"
start "" .venv\Scripts\pythonw.exe -m app.main
