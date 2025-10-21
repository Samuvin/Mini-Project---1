#!/bin/bash

# Parkinson's Disease Detection System - Start Script
# This script handles everything: setup, installation, training, and server start

echo "============================================================"
echo "Parkinson's Disease Detection System"
echo "Automated Setup & Start"
echo "============================================================"
echo ""

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored messages
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Step 1: Check Python
print_info "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi
PYTHON_VERSION=$(python3 --version)
print_success "Found $PYTHON_VERSION"
echo ""

# Step 2: Create/Activate Virtual Environment
print_info "Setting up virtual environment..."
if [ ! -d "venv" ]; then
    print_info "Creating new virtual environment..."
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_success "Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate
print_success "Virtual environment activated"
echo ""

# Step 3: Install/Update Requirements
print_info "Checking dependencies..."
pip install --upgrade pip -q
print_info "Installing required packages (this may take a few minutes)..."
pip install -r requirements.txt -q

if [ $? -eq 0 ]; then
    print_success "All dependencies installed"
else
    print_error "Failed to install dependencies"
    exit 1
fi
echo ""

# Step 4: Check and Download Datasets
print_info "Checking datasets..."

# Check speech data (only dataset we need for speech-only model)
if [ ! -f "data/raw/speech/parkinsons.csv" ]; then
    print_info "Speech dataset not found. Downloading..."
    python download_datasets.py
    if [ $? -eq 0 ]; then
        print_success "Speech dataset downloaded"
    else
        print_error "Failed to download speech dataset"
        print_error "Cannot proceed without speech data"
        exit 1
    fi
else
    print_success "Speech dataset found"
fi
echo ""

# Step 5: Generate Multi-Modal Datasets
print_info "Checking multi-modal datasets..."
if [ ! -f "data/raw/handwriting/handwriting_data.csv" ] || [ ! -f "data/raw/gait/gait_data.csv" ]; then
    print_info "Generating handwriting and gait datasets..."
    python generate_modality_datasets.py
    if [ $? -eq 0 ]; then
        print_success "Multi-modal datasets generated"
    else
        print_error "Failed to generate datasets"
        exit 1
    fi
else
    print_success "Multi-modal datasets found"
fi
echo ""

# Step 6: Check and Train Models
print_info "Checking model status..."

# Check Speech Model
if [ ! -f "models/speech_model.joblib" ] || [ ! -f "models/speech_scaler.joblib" ]; then
    if [ ! -f "models/best_model.joblib" ]; then
        print_info "Speech model not found. Training..."
        python train.py
        if [ $? -eq 0 ]; then
            # Copy to speech_model naming convention
            cp models/best_model.joblib models/speech_model.joblib
            cp models/scaler.joblib models/speech_scaler.joblib
            print_success "Speech model trained"
        else
            print_error "Speech model training failed"
            exit 1
        fi
    else
        # Copy existing model to new naming convention
        cp models/best_model.joblib models/speech_model.joblib
        cp models/scaler.joblib models/speech_scaler.joblib
        print_success "Speech model ready"
    fi
else
    print_success "Speech model found"
fi

# Check Handwriting Model
if [ ! -f "models/handwriting_model.joblib" ] || [ ! -f "models/handwriting_scaler.joblib" ]; then
    print_info "Handwriting model not found. Training..."
    python train_handwriting_model.py
    if [ $? -eq 0 ]; then
        print_success "Handwriting model trained"
    else
        print_error "Handwriting model training failed"
        exit 1
    fi
else
    print_success "Handwriting model found"
fi

# Check Gait Model
if [ ! -f "models/gait_model.joblib" ] || [ ! -f "models/gait_scaler.joblib" ]; then
    print_info "Gait model not found. Training..."
    python train_gait_model.py
    if [ $? -eq 0 ]; then
        print_success "Gait model trained"
    else
        print_error "Gait model training failed"
        exit 1
    fi
else
    print_success "Gait model found"
fi

print_success "All models ready"
echo ""

# Step 7: Stop any existing server
print_info "Checking for running server..."
pkill -f "gunicorn.*wsgi:app" 2>/dev/null
if [ $? -eq 0 ]; then
    print_info "Stopped existing server"
    sleep 2
fi
echo ""

# Step 8: Start the Server
print_success "Starting server..."
echo ""
echo "============================================================"
echo "Server Configuration:"
echo "  • URL: http://localhost:8000"
echo "  • Workers: 2"
echo "  • Mode: Production (Gunicorn)"
echo "============================================================"
echo ""
print_info "Server is starting..."
echo "Press Ctrl+C to stop the server"
echo ""

# Start Gunicorn in foreground
gunicorn --bind 0.0.0.0:8000 \
         --workers 2 \
         --timeout 120 \
         --access-logfile - \
         --error-logfile - \
         wsgi:app

# If we get here, server was stopped
echo ""
print_info "Server stopped"

