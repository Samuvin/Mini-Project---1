# Parkinson's Disease Early Prediction System

A multimodal deep learning system for early prediction of Parkinson's Disease using **SE-ResNet with Attention Fusion** across speech, handwriting, and gait modalities.

## Project Overview

Parkinson's Disease is the second most common neurodegenerative disorder, causing tremors, stiffness, and slow movement. Early prediction is crucial for improving patient outcomes. This project uses a deep learning framework that analyzes **three data modalities** with explainable AI (Grad-CAM) to provide accurate and interpretable predictions.

## Methodology

- **Architecture**: SE-ResNet 1D + Attention Fusion (multimodal deep learning)
- **Modalities**: Speech (22 features), Handwriting (10 features), Gait (10 features)
- **Explainability**: Grad-CAM feature importance, attention weights, SE channel weights
- **Class Balancing**: SMOTE oversampling
- **Framework**: PyTorch
- **Fallback**: sklearn ensemble (SVM + LR) when DL model is not available
- **Deployment**: Flask + Gunicorn WSGI server with JWT authentication

## Datasets

This project uses three publicly available real datasets:

### 1. Speech Data - UCI Parkinson's Dataset
- **Source**: https://archive.ics.uci.edu/ml/datasets/Parkinsons
- **Features**: 22 acoustic measurements (jitter, shimmer, HNR, pitch, nonlinear dynamics)
- **Samples**: 195 (147 PD, 48 healthy)

### 2. Handwriting Data - PaHaW / NewHandPD
- **Features**: 10 kinematic measurements (pressure, velocity, tremor, etc.)
- **Format**: Pre-extracted features in CSV

### 3. Gait Data - PhysioNet Database
- **Source**: https://physionet.org/content/gaitpdb/1.0.0/
- **Features**: 10 temporal-spatial parameters (stride, cadence, speed, asymmetry)
- **Format**: Processed stride data in CSV

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

```bash
cd fn

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment variables
cp .env.example .env
# Edit .env with your MONGODB_URI and JWT_SECRET_KEY
```

## Project Structure

```
fn/
├── dl_models/            # Deep learning modules
│   ├── networks.py       # SE-ResNet + Attention Fusion architecture
│   ├── dataset.py        # PyTorch dataset for multimodal data
│   ├── trainer.py        # Training loop with early stopping
│   ├── inference.py      # DL predictor for production inference
│   └── gradcam.py        # Grad-CAM explainability
├── src/                  # sklearn fallback and utilities
│   ├── core/             # Model manager, predictor
│   ├── data/             # Data loading and preprocessing
│   ├── models/           # sklearn model implementations
│   ├── evaluation/       # Evaluation metrics
│   └── utils/            # Configuration utilities
├── webapp/               # Flask web application
│   ├── app.py            # Application factory
│   ├── api/              # REST API (predict, auth, upload)
│   ├── middleware/        # JWT authentication middleware
│   ├── models/           # User model (MongoDB)
│   ├── templates/        # Jinja2 HTML templates
│   └── static/           # CSS, JS, images
├── models/               # Saved trained models (.joblib, .pt)
├── data/                 # Datasets
├── train.py              # sklearn training pipeline
├── train_dl.py           # Deep learning training pipeline
├── wsgi.py               # WSGI entry point
├── gunicorn_config.py    # Gunicorn configuration
├── config.yaml           # Hyperparameters and paths
└── requirements.txt      # Python dependencies
```

## Usage

### 1. Train the Deep Learning Model

```bash
python train_dl.py
```

This trains the SE-ResNet + Attention Fusion model and saves:
- Model weights to `models/multimodal_pd_net.pt`
- Feature scalers to `models/dl_*_scaler.joblib`
- Training plots and metrics

### 2. Run the Application

```bash
./start.sh
```

Or manually:

```bash
gunicorn -c gunicorn_config.py wsgi:app
```

Visit `http://localhost:8000` in your browser.

### 3. Stop the Application

```bash
./stop.sh
```

## Architecture

### SE-ResNet 1D + Attention Fusion

Each modality is processed by its own SE-ResNet branch:
1. **1D Convolution** - Extracts local patterns from feature vectors
2. **Residual SE Blocks** - Skip connections + Squeeze-and-Excitation channel attention
3. **Attention Fusion** - Learned weights combine modality embeddings
4. **Dense Classifier** - Final prediction with dropout regularization

### Explainability

- **Grad-CAM**: Per-feature importance scores showing which inputs drive the prediction
- **Attention Weights**: How much each modality (speech, handwriting, gait) contributes
- **SE Channel Weights**: Internal channel attention within each modality branch

## API Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/health` | GET | No | Health check and model status |
| `/api/predict` | POST | Yes | Single prediction |
| `/api/predict_batch` | POST | Yes | Batch predictions |
| `/api/model_info` | GET | Yes | Model information |
| `/api/auth/register` | POST | No | User registration |
| `/api/auth/login` | POST | No | User login |
| `/api/auth/logout` | POST | Yes | User logout |
| `/api/upload/audio` | POST | Yes | Upload audio for speech features |
| `/api/upload/handwriting` | POST | Yes | Upload image for handwriting features |
| `/api/upload/gait` | POST | Yes | Upload video for gait features |

## Configuration

Edit `config.yaml` to adjust:
- Deep learning hyperparameters (learning rate, epochs, architecture)
- Data split ratios
- Feature extraction parameters
- Server settings

Environment variables (`.env`):
- `MONGODB_URI` - MongoDB connection string
- `JWT_SECRET_KEY` - Secret key for JWT token signing

## Testing

```bash
pytest tests/
pytest tests/ --cov=src --cov-report=html
```

## Tech Stack

- **Backend**: Flask, Gunicorn, PyTorch
- **Frontend**: Jinja2, Bootstrap 5, Chart.js
- **Database**: MongoDB (user auth)
- **Auth**: JWT (PyJWT + bcrypt)
- **ML Fallback**: scikit-learn, XGBoost, LightGBM

---

**Disclaimer**: This system is for research and educational purposes only. It is not intended for clinical diagnosis. Always consult healthcare professionals for medical advice.
