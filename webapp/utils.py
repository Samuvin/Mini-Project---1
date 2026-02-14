"""Helper module to load and display model metrics."""

import json
import logging
import joblib
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def get_model_metrics(models_dir: Path) -> Dict[str, any]:
    """
    Get model performance metrics from saved model or metrics file.
    
    Args:
        models_dir: Path to models directory
        
    Returns:
        Dictionary with model metrics
    """
    # Try to load metrics from JSON file first
    metrics_file = models_dir / 'model_metrics.json'
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            return json.load(f)
    
    # Default metrics (fallback)
    return {
        'svm': {
            'accuracy': 0.90,
            'precision': 1.00,
            'recall': 0.87,
            'f1_score': 0.93,
            'roc_auc': 0.95,
            'model_type': 'SVM (RBF)',
            'available': False
        },
        'logistic_regression': {
            'accuracy': 0.85,
            'precision': 0.83,
            'recall': 0.81,
            'f1_score': 0.82,
            'roc_auc': 0.87,
            'model_type': 'Logistic Regression',
            'available': False
        }
    }


def save_model_metrics(metrics: Dict[str, any], models_dir: Path) -> None:
    """
    Save model metrics to JSON file.
    
    Args:
        metrics: Dictionary with model metrics
        models_dir: Path to models directory
    """
    metrics_file = models_dir / 'model_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to: %s", metrics_file)


def check_model_files(models_dir: Path) -> Dict[str, bool]:
    """
    Check which model files are available.
    
    Args:
        models_dir: Path to models directory
        
    Returns:
        Dictionary indicating which files exist
    """
    return {
        'svm_model': (models_dir / 'svm_model.joblib').exists(),
        'lr_model': (models_dir / 'logistic_regression_model.joblib').exists(),
        'best_model': (models_dir / 'best_model.joblib').exists(),
        'svm_roc': (models_dir / 'svm_roc_curve.png').exists(),
        'lr_roc': (models_dir / 'lr_roc_curve.png').exists(),
        'svm_cm': (models_dir / 'svm_confusion_matrix.png').exists(),
        'lr_cm': (models_dir / 'lr_confusion_matrix.png').exists(),
    }

