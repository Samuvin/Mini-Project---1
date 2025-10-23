"""
Model adapters implementing Adapter pattern.
Wraps sklearn/joblib models to match our interfaces.
"""

import numpy as np
from typing import Any
from src.core.interfaces import IPredictor, IFeatureScaler


class SklearnModelAdapter(IPredictor):
    """Adapter for sklearn-compatible models."""
    
    def __init__(self, model: Any):
        """
        Initialize with a sklearn model.
        
        Args:
            model: Any sklearn-compatible model with predict/predict_proba
        """
        self._model = model
    
    def predict(self, features: np.ndarray) -> int:
        """Make prediction."""
        return int(self._model.predict(features)[0])
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        return self._model.predict_proba(features)[0]


class SklearnScalerAdapter(IFeatureScaler):
    """Adapter for sklearn scalers."""
    
    def __init__(self, scaler: Any):
        """
        Initialize with a sklearn scaler.
        
        Args:
            scaler: Any sklearn-compatible scaler
        """
        self._scaler = scaler
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """Transform features."""
        return self._scaler.transform(features)

