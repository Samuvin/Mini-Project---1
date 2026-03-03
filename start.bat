@echo off
REM Parkinson's Disease Prediction System - Start Script (Light mode only)
REM Uses custom logic only; no ML libraries

echo ============================================================
echo Parkinson's Disease Prediction System
echo Automated Setup ^& Start (Light mode - Windows)
echo ============================================================
echo.

cd /d "%~dp0"

REM Step 1: Check Python
echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed. Please install Python 3.8 or higher.
    exit /b 1
)
python --version
echo.

REM Step 2: Create/Activate Virtual Environment
echo Setting up virtual environment...
if not exist "venv" (
    echo Creating new virtual environment...
    python -m venv venv
    echo Virtual environment created
) else (
    echo Virtual environment already exists
)

call venv\Scripts\activate.bat
echo Virtual environment activated
echo.

REM Step 3: Install light dependencies only
echo Installing dependencies (light mode)...
python -m pip install --upgrade pip -q
pip install -r requirements-light.txt -q

if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    exit /b 1
)
echo Dependencies installed
echo.

REM Step 4: Start the server (light mode)
set USE_LIGHT_MODE=1
set PORT=8000
echo Starting server (light mode - custom logic only)...
echo.
echo ============================================================
echo   URL: http://localhost:8000
echo   Mode: Light (custom logic; no ML libraries)
echo ============================================================
echo.
echo Press Ctrl+C to stop the server
echo.

python wsgi.py

echo.
echo Server stopped
