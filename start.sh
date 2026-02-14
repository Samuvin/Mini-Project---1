#!/bin/bash

# Parkinson's Disease Prediction System - Start Script
# Handles setup, installation, dataset generation, training, and server start

echo "============================================================"
echo "Parkinson's Disease Prediction System"
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

# Step 4: Check datasets
print_info "Checking datasets..."

if [ ! -f "data/raw/speech/parkinsons.csv" ]; then
    print_error "Speech dataset not found at data/raw/speech/parkinsons.csv"
    print_error "Please ensure the UCI Parkinson's dataset is available."
    exit 1
else
    print_success "Speech dataset found"
fi

# Step 5: Generate multimodal datasets if missing
print_info "Checking multimodal datasets..."
if [ ! -f "data/raw/handwriting/handwriting_data.csv" ] || [ ! -f "data/raw/gait/gait_data.csv" ]; then
    print_info "Generating handwriting and gait datasets..."
    python generate_modality_datasets.py
    if [ $? -eq 0 ]; then
        print_success "Multimodal datasets generated"
    else
        print_error "Failed to generate datasets"
        exit 1
    fi
else
    print_success "Multimodal datasets found"
fi
echo ""

# Step 6: Check and train sklearn fallback models
print_info "Checking sklearn fallback models..."

if [ ! -f "models/speech_model.joblib" ] || [ ! -f "models/speech_scaler.joblib" ]; then
    if [ ! -f "models/best_model.joblib" ]; then
        print_info "Training sklearn speech model..."
        python train.py
        if [ $? -eq 0 ]; then
            cp models/best_model.joblib models/speech_model.joblib
            cp models/scaler.joblib models/speech_scaler.joblib
            print_success "sklearn speech model trained"
        else
            print_error "sklearn speech model training failed"
            exit 1
        fi
    else
        cp models/best_model.joblib models/speech_model.joblib
        cp models/scaler.joblib models/speech_scaler.joblib
        print_success "sklearn speech model ready"
    fi
else
    print_success "sklearn speech model found"
fi

# Step 7: Check DL model (optional - app falls back to sklearn if missing)
if [ -f "models/multimodal_pd_net.pt" ]; then
    print_success "Deep learning model found (SE-ResNet + Attention Fusion)"
else
    print_info "Deep learning model not found. Using sklearn fallback."
    print_info "To train the DL model, run: python train_dl.py"
fi

print_success "All models ready"
echo ""

# Step 8: Stop any existing server
print_info "Checking for running server..."
pkill -f "gunicorn.*wsgi:app" 2>/dev/null
if [ $? -eq 0 ]; then
    print_info "Stopped existing server"
    sleep 2
fi
echo ""

# Step 9: Start the Server
print_success "Starting server..."
echo ""
echo "============================================================"
echo "Server Configuration:"
echo "  • URL: http://localhost:8000"
echo "  • Workers: 2"
echo "  • Mode: Production (Gunicorn)"
echo "  • Backend: SE-ResNet + Attention Fusion (DL) / sklearn fallback"
echo "============================================================"
echo ""
print_info "Server is starting..."
echo "Press Ctrl+C to stop the server"
echo ""

gunicorn --bind 0.0.0.0:8000 \
         --workers 2 \
         --timeout 120 \
         --access-logfile - \
         --error-logfile - \
         wsgi:app

echo ""
print_info "Server stopped"
