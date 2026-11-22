@echo off
chcp 65001 >nul
REM ===== تشغيل النظام =====
REM أمثلة:  run.bat
REM         run.bat --source http://192.168.1.5:8080/video
REM         run.bat --profile patients\ahmed.json
cd /d "%~dp0"
if not exist venv\Scripts\python.exe (
    echo البيئة غير مثبّتة. شغّل setup.bat أولاً.
    pause
    exit /b 1
)
set TF_CPP_MIN_LOG_LEVEL=2
venv\Scripts\python.exe main.py %*
if errorlevel 1 pause
