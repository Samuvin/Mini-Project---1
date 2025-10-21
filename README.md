# Parkinson's Disease Early Detection System

A machine learning system for early detection of Parkinson's Disease using **multimodal data analysis** (speech, handwriting, and gait patterns) with Support Vector Machines and kernel optimization.

## 🎯 Project Overview

Parkinson's Disease is the second most common brain disorder, causing tremors, stiffness, and slow movement. Early detection is crucial for improving patient outcomes. This project uses advanced machine learning techniques to analyze **multiple data modalities** to provide accurate and reliable early detection.

## 🔬 Methodology

- **Approach**: Multimodal data fusion (early fusion strategy)
- **Datasets**: REAL patient data only - UCI Parkinson's, PaHaW Handwriting, PhysioNet Gait
- **Features**: ~42 combined features from three modalities
- **Primary Model**: Support Vector Machine (SVM) with kernel optimization (RBF, Polynomial, Linear, Sigmoid)
- **Optimization**: GridSearchCV with cross-validation
- **Evaluation**: Focus on recall to minimize false negatives
- **Deployment**: Gunicorn WSGI server for reliable operation

## 📊 Datasets

This project uses THREE publicly available REAL datasets:

### 1. Speech Data - UCI Parkinson's Dataset
- **Source**: https://archive.ics.uci.edu/ml/datasets/Parkinsons
- **Features**: 22 acoustic measurements
- **Samples**: 195 (147 PD, 48 healthy)
- **Status**: ✅ Automatically downloaded

### 2. Handwriting Data - PaHaW / NewHandPD
- **Features**: 10 kinematic measurements (pressure, velocity, tremor, etc.)
- **Format**: Pre-extracted features in CSV
- **Status**: ⚠️ Requires manual download (see DATASETS.md)

### 3. Gait Data - PhysioNet Database  
- **Source**: https://physionet.org/content/gaitpdb/1.0.0/
- **Features**: 10 temporal-spatial parameters
- **Format**: Processed stride data in CSV
- **Status**: ⚠️ Requires download and processing (see DATASETS.md)

**⚠️ IMPORTANT**: This system uses **ONLY REAL DATASETS** - **NO synthetic data generation whatsoever**.

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

```bash
# Navigate to the project directory
cd fn

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📁 Project Structure

```
fn/
├── data/                   # Data storage
│   ├── raw/               # Original datasets
│   │   ├── speech/        # UCI Parkinson's dataset
│   │   ├── handwriting/   # PaHaW features
│   │   └── gait/          # PhysioNet features
│   └── processed/         # Preprocessed data
├── src/                   # Source code
│   ├── data/             # Data loading and preprocessing
│   ├── features/         # Feature extraction (speech, handwriting, gait)
│   ├── models/           # Model implementations (LR, SVM)
│   ├── evaluation/       # Evaluation metrics
│   └── utils/            # Utility functions
├── webapp/               # Web application
│   ├── app.py           # Flask application
│   ├── templates/       # HTML templates
│   ├── static/          # CSS, JS, images
│   └── api/             # API endpoints
├── models/              # Saved trained models
├── tests/              # Unit tests
├── train.py            # Training pipeline
├── wsgi.py            # WSGI entry point
├── gunicorn_config.py # Gunicorn configuration
├── DATASETS.md        # Dataset documentation
└── README.md          # This file
```

## 💻 Usage

### 1. Download Datasets

**Speech data** is downloaded automatically. For handwriting and gait data:

```bash
# Check dataset status
python -m src.data.data_loader

# See DATASETS.md for detailed download instructions
```

**Important**: You must obtain real handwriting and gait datasets before training. See `DATASETS.md` for detailed instructions.

### 2. Train Model

Train the multimodal SVM model:

```bash
python train.py
```

This will:
- Load all three datasets (or fallback to speech-only if others unavailable)
- Preprocess and split data
- Train Logistic Regression baseline
- Train SVM with kernel optimization
- Save the best model to `models/`

### 3. Run Application

Start the application using Gunicorn:

```bash
chmod +x start_server.sh
./start_server.sh
```

Or manually:

```bash
gunicorn -c gunicorn_config.py wsgi:app
```

Visit `http://localhost:8000` in your browser.

## 🎯 Model Performance

### Expected Results (Multimodal)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression (Baseline) | ~85-88% | ~86% | ~84% | ~85% | ~0.90 |
| SVM (RBF - Optimized) | ~92-95% | ~93% | ~94% | ~93% | ~0.97 |

*Multimodal fusion typically achieves 3-5% better performance than single modality*

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest tests/

# Run with coverage report
pytest tests/ --cov=src --cov-report=html
```

## 🌐 Web Application Features

- **Multimodal Input**: Enter features from speech, handwriting, and gait
- **Manual Entry**: Text input for all 42 features
- **CSV Upload**: Upload file with all features
- **Example Data**: Load sample data from real datasets
- **Real-time Prediction**: Immediate classification results
- **Confidence Scores**: Probability estimates for predictions
- **Feature Reference**: Complete guide with accordions for each modality

## 🔧 Configuration

Modify `config.yaml` to adjust:
- Data split ratios
- Model hyperparameters
- Feature extraction parameters
- Server settings

## 📝 Key Features

- ✅ Multimodal data fusion (speech + handwriting + gait)
- ✅ ONLY real datasets - NO synthetic data generation
- ✅ 42 total features across three modalities
- ✅ Multiple kernel support (Linear, RBF, Polynomial, Sigmoid)
- ✅ Automated hyperparameter optimization
- ✅ Class imbalance handling with SMOTE
- ✅ Comprehensive evaluation metrics
- ✅ Cross-validation for robust performance
- ✅ User-friendly multimodal web interface
- ✅ Model persistence and versioning
- ✅ Gunicorn WSGI server for deployment

## 📖 Feature Descriptions

### Speech Features (22)
Acoustic measurements including jitter, shimmer, HNR, pitch variations, and nonlinear dynamics.

### Handwriting Features (10)
Kinematic measures including pressure, velocity, acceleration, tremor frequency, and fluency.

### Gait Features (10)
Temporal-spatial parameters including stride intervals, cadence, gait speed, and asymmetry.

See `DATASETS.md` for complete feature descriptions.

## 🎓 Research References

This project uses data from:
- **UCI Machine Learning Repository** - Parkinson's Dataset
- **PaHaW Database** - Parkinson's Handwriting
- **PhysioNet** - Gait in Parkinson's Disease Database

See `DATASETS.md` for citations.

## 📄 License

This project is for educational and research purposes. Please cite the original dataset sources when using this code.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 🔮 Future Improvements

- Deep learning models (CNN, LSTM) for feature extraction
- Late fusion strategies for multimodal data
- Real-time audio and handwriting capture
- Mobile application development
- Integration with telemedicine platforms

## 📧 Contact

For questions or collaboration opportunities, please open an issue on GitHub.

---

**Disclaimer**: This system is for research and educational purposes only. It is not intended for clinical diagnosis. Always consult healthcare professionals for medical advice.

**No Synthetic Data**: This system uses ONLY real patient datasets. There is NO synthetic data generation in this codebase.
