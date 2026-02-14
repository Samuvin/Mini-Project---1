@echo off
REM Parkinson's Disease Prediction System - Start Script for Windows
REM Handles setup, installation, dataset generation, training, and server start

echo ============================================================
echo Parkinson's Disease Prediction System
echo Automated Setup ^& Start (Windows)
echo ============================================================
echo.

REM Get the directory where the script is located
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

REM Step 3: Install/Update Requirements
echo Checking dependencies...
python -m pip install --upgrade pip -q
echo Installing required packages (this may take a few minutes)...
pip install -r requirements.txt -q

if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    exit /b 1
)
echo All dependencies installed
echo.

REM Step 4: Check datasets
echo Checking datasets...
if not exist "data\raw\speech\parkinsons.csv" (
    echo ERROR: Speech dataset not found at data\raw\speech\parkinsons.csv
    echo Please ensure the UCI Parkinson's dataset is available.
    exit /b 1
)
echo Speech dataset found

REM Step 5: Generate multimodal datasets if missing
echo Checking multimodal datasets...
if not exist "data\raw\handwriting\handwriting_data.csv" (
    echo Generating handwriting and gait datasets...
    python generate_modality_datasets.py
    if errorlevel 1 (
        echo ERROR: Failed to generate datasets
        exit /b 1
    )
    echo Multimodal datasets generated
) else (
    echo Multimodal datasets found
)
echo.

REM Step 6: Check and train sklearn fallback models
echo Checking sklearn fallback models...
if not exist "models\speech_model.joblib" (
    if not exist "models\best_model.joblib" (
        echo Training sklearn speech model...
        python train.py
        if errorlevel 1 (
            echo ERROR: sklearn speech model training failed
            exit /b 1
        )
        copy models\best_model.joblib models\speech_model.joblib >nul
        copy models\scaler.joblib models\speech_scaler.joblib >nul
        echo sklearn speech model trained
    ) else (
        copy models\best_model.joblib models\speech_model.joblib >nul
        copy models\scaler.joblib models\speech_scaler.joblib >nul
        echo sklearn speech model ready
    )
) else (
    echo sklearn speech model found
)

REM Step 7: Check DL model (optional)
if exist "models\multimodal_pd_net.pt" (
    echo Deep learning model found (SE-ResNet + Attention Fusion)
) else (
    echo Deep learning model not found. Using sklearn fallback.
    echo To train the DL model, run: python train_dl.py
)

echo All models ready
echo.

REM Step 8: Stop any existing server
echo Checking for running server...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq *wsgi*" >nul 2>&1
echo.

REM Step 9: Start the Server (using Flask dev server for Windows)
echo Starting server...
echo.
echo ============================================================
echo Server Configuration:
echo   URL: http://localhost:8000
echo   Mode: Development (Flask - Windows compatible)
echo   Backend: SE-ResNet + Attention Fusion (DL) / sklearn fallback
echo ============================================================
echo.
echo Server is starting...
echo Press Ctrl+C to stop the server
echo.

REM Use Flask's development server on Windows (gunicorn doesn't work)
set FLASK_APP=wsgi:app
set FLASK_ENV=development
set PORT=8000
python -m flask run --host=0.0.0.0 --port=8000

echo.
echo Server stopped
