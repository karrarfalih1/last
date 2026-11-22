@echo off
chcp 65001 >nul
REM ===== تثبيت المشروع على جهاز جديد (مرة واحدة) =====
REM يحتاج: Python مثبّت (أي إصدار)، واتصال إنترنت.
cd /d "%~dp0"

echo [1/4] تثبيت uv (مدير بيئات Python سريع)...
python -m pip install -q uv || goto :error

echo [2/4] إنشاء بيئة Python 3.11 داخل المشروع...
if not exist venv\Scripts\python.exe (
    python -m uv venv --python 3.11 venv || goto :error
)

echo [3/4] تثبيت المكتبات...
python -m uv pip install --python venv\Scripts\python.exe -r requirements.txt || goto :error

echo [4/4] نموذج تتبع الوجه والملفات الصوتية...
if not exist models\face_landmarker.task (
    mkdir models 2>nul
    curl -sSL -o models\face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task || goto :error
)
venv\Scripts\python.exe generate_audio.py

echo.
echo تم التثبيت. للتشغيل: run.bat
if not exist .env echo تذكير: لربط Blynk أنشئ ملف .env فيه السطر  BLYNK_TOKEN=...
pause
exit /b 0

:error
echo فشل التثبيت. راجع الرسالة أعلاه.
pause
exit /b 1
