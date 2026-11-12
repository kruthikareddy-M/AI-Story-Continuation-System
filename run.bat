@echo off
setlocal
title AI Story Continuation System

cd /d "%~dp0"

if exist "story-continuation\run_app.py" (
    cd story-continuation
)

echo ==============================================================
echo Starting AI Story Continuation System...
echo ==============================================================

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" run_app.py
) else (
    python run_app.py
)

pause
