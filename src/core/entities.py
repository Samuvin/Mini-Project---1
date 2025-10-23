"""
Domain entities representing core business objects.
Following Single Responsibility Principle (SRP).
"""

from dataclasses import dataclass
from typing import Dict, Optional, List
import numpy as np


@dataclass
class PredictionResult:
    """Immutable prediction result entity."""
    
    prediction: int
    prediction_label: str
    confidence: float
    probabilities: Dict[str, float]
    modality: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            'prediction': self.prediction,
            'prediction_label': self.prediction_label,
            'confidence': self.confidence,
            'probabilities': self.probabilities,
            'modality': self.modality
        }


@dataclass
class EnsemblePredictionResult:
    """Result from ensemble prediction."""
    
    prediction: int
    prediction_label: str
    confidence: float
    probabilities: Dict[str, float]
    modalities_used: List[str]
    individual_predictions: Dict[str, PredictionResult]
    ensemble_method: str
    success: bool = True
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            'success': self.success,
            'prediction': self.prediction,
            'prediction_label': self.prediction_label,
            'confidence': self.confidence,
            'probabilities': self.probabilities,
            'modalities_used': self.modalities_used,
            'individual_predictions': {
                k: v.to_dict() for k, v in self.individual_predictions.items()
            },
            'ensemble_method': self.ensemble_method
        }


@dataclass
class FeatureVector:
    """Feature vector with metadata."""
    
    modality: str
    features: np.ndarray
    feature_names: Optional[List[str]] = None
    
    def __post_init__(self):
        """Validate feature vector."""
        if not isinstance(self.features, np.ndarray):
            self.features = np.array(self.features)
        
        if self.features.ndim == 1:
            self.features = self.features.reshape(1, -1)


@dataclass
class CalibrationContext:
    """Context for calibration operations."""
    
    ground_truth_hint: Optional[str] = None
    temperature: float = 1.0
    method: str = 'temperature_scaling'
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'ground_truth_hint': self.ground_truth_hint,
            'temperature': self.temperature,
            'method': self.method
        }

