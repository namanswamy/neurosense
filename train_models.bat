@echo off
title NeuroSense — ML Model Training
echo.
echo ============================================================
echo   NeuroSense — One-Click ML Setup
echo ============================================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    echo         Download from https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

echo [1/3] Installing pip (if missing) and Python dependencies...
python -m ensurepip --upgrade >nul 2>&1
python -m pip install --upgrade pip --quiet >nul 2>&1
python -m pip install pandas numpy scikit-learn --quiet
if errorlevel 1 (
    echo [ERROR] pip install failed.
    pause
    exit /b 1
)
echo      Done.
echo.

echo [2/3] Training models (this takes ~30 seconds)...
echo.
python "%~dp0ml\setup_and_train.py"
if errorlevel 1 (
    echo.
    echo [ERROR] Training failed. See errors above.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   Ready! Start your server with:  node server.js
echo ============================================================
echo.
pause
