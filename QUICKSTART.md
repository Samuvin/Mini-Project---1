# Quick Start Guide

Welcome to the Parkinson's Disease Detection System! This guide will help you get started quickly.

## Prerequisites

- Python 3.8 or higher
- pip package manager
- At least 2GB of free disk space

## Installation

### 1. Set up Python environment

```bash
# Navigate to the project directory
cd /Users/jenishs/Desktop/Spryzen/fn

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
# venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

This will install all required packages including:
- Machine learning libraries (scikit-learn, imbalanced-learn)
- Data processing (pandas, numpy, scipy)
- Visualization (matplotlib, seaborn, plotly)
- Web framework (Flask)
- Jupyter notebooks

## Quick Start (5 minutes)

### Option 1: Train Models and Run Web App

```bash
# 1. Download and prepare datasets
python src/data/data_loader.py --download

# 2. Train models (this may take 10-15 minutes)
python train.py

# 3. Run the web application
python webapp/app.py

# 4. Open your browser and go to:
http://localhost:5000
```

### Option 2: Use Pre-existing Models (if available)

```bash
# If models are already trained, just run the webapp:
python webapp/app.py
```

## Project Structure

```
fn/
├── data/              # Data storage
├── notebooks/         # Jupyter notebooks for exploration
├── src/              # Source code
│   ├── data/         # Data loading and preprocessing
│   ├── features/     # Feature extraction
│   ├── models/       # ML model implementations
│   ├── evaluation/   # Evaluation metrics
│   └── utils/        # Utilities
├── webapp/           # Web application
├── models/           # Saved trained models
├── tests/            # Test suite
├── config.yaml       # Configuration
├── train.py          # Training script
└── requirements.txt  # Dependencies
```

## Usage Examples

### 1. Training Models

```bash
# Train all models with default settings
python train.py

# This will:
# - Load and preprocess data
# - Train Logistic Regression baseline
# - Train SVM with kernel optimization
# - Evaluate both models
# - Save the best model
```

### 2. Using the Web Interface

1. Start the web server:
   ```bash
   python webapp/app.py
   ```

2. Navigate to `http://localhost:5000`

3. Go to the "Predict" page

4. Choose input method:
   - **Manual Entry**: Enter feature values separated by commas
   - **Upload CSV**: Upload a CSV file with features
   - **Example Data**: Load sample data for testing

5. Click "Make Prediction" to get results

### 3. Using the API

```python
import requests
import json

# Prepare data
url = "http://localhost:5000/api/predict"
data = {
    "features": [0.5, 1.2, -0.3, 0.8, ...]  # Your feature values
}

# Make request
response = requests.post(url, json=data)
result = response.json()

# Print results
print(f"Prediction: {result['prediction_label']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### 4. Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_data.py -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html
```

### 5. Using Jupyter Notebooks

```bash
# Start Jupyter
jupyter notebook

# Open notebooks in the notebooks/ directory:
# - 01_data_exploration.ipynb
# - 02_feature_engineering.ipynb
# - 03_baseline_logistic_regression.ipynb
# - 04_svm_kernel_optimization.ipynb
# - 05_model_evaluation.ipynb
```

## Configuration

Edit `config.yaml` to customize:

```yaml
# Data configuration
data:
  train_size: 0.70
  val_size: 0.15
  test_size: 0.15

# Model parameters
models:
  svm:
    kernel: ['linear', 'rbf', 'poly', 'sigmoid']
    C: [0.1, 1, 10, 100]
    gamma: ['scale', 'auto', 0.001, 0.01, 0.1, 1]

# Training settings
training:
  cv_folds: 5
  scoring: 'roc_auc'
  use_smote: true

# Web app settings
webapp:
  host: '0.0.0.0'
  port: 5000
  debug: true
```

## Troubleshooting

### Issue: "Model not loaded"
**Solution**: Run `python train.py` first to train models

### Issue: "Data files not found"
**Solution**: Run `python src/data/data_loader.py --download` to download datasets

### Issue: Port 5000 already in use
**Solution**: Change the port in `config.yaml` under `webapp.port`

### Issue: Import errors
**Solution**: Make sure virtual environment is activated and dependencies are installed

## Performance Expectations

### Model Training Time
- Logistic Regression: 1-2 minutes
- SVM with GridSearch: 8-12 minutes
- Total training time: ~15 minutes

### Expected Performance
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 82-85% | ~83% | ~81% | ~82% | ~0.87 |
| SVM (RBF Optimized) | 90-93% | ~91% | ~92% | ~91% | ~0.95 |

## Next Steps

1. **Explore the data**: Open `notebooks/01_data_exploration.ipynb`
2. **Customize models**: Modify hyperparameters in `config.yaml`
3. **Try predictions**: Use the web interface at `http://localhost:5000`
4. **Review API**: Check `/documentation` page for API usage
5. **Run tests**: Execute `pytest tests/ -v` to verify everything works

## Support

For issues, questions, or contributions:
1. Check the main README.md for detailed documentation
2. Review the code documentation in source files
3. Run tests to verify your setup: `pytest tests/ -v`

## Important Disclaimer

This system is for **research and educational purposes only**. It is not intended for clinical diagnosis. Always consult healthcare professionals for medical advice.

---

**Congratulations!** You're now ready to use the Parkinson's Disease Detection System. Start with `python train.py` to train your models, then launch the web app with `python webapp/app.py`.

