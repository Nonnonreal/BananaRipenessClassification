@echo off
title BananaVision AI WebApp
echo ========================================================
echo   BananaVision AI - Video Detection ^& Classification
echo ========================================================
echo.

cd /d "%~dp0"

if not exist ".env" (
    echo [.env not found, copying from .env.example...]
    copy .env.example .env
)

if exist ".venv\Scripts\python.exe" (
    echo [Using virtual environment in .venv]
    .venv\Scripts\python.exe run.py
) else (
    echo [Using system py launcher]
    py run.py
)

pause
