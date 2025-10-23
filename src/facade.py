"""
Facade pattern for backward compatibility.
Provides simple interface to complex subsystem (Facade Pattern).
"""

import numpy as np
from typing import Dict, List, Optional, Any
from src.container import get_prediction_service
from src.core.interfaces import IPredictionService


class PredictionFacade:
    """
    Simplified facade for prediction operations.
    
    Implements Facade pattern to provide backward compatibility
    while using the new SOLID-compliant architecture underneath.
    """
    
    def __init__(self):
        """Initialize facade with prediction service."""
        self._service: IPredictionService = get_prediction_service()
    
    def predict_ensemble(
        self,
        speech_features: Optional[np.ndarray] = None,
        handwriting_features: Optional[np.ndarray] = None,
        gait_features: Optional[np.ndarray] = None,
        voting_method: str = 'soft',
        calibration_context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Make ensemble prediction (backward-compatible API).
        
        Args:
            speech_features: Speech feature array
            handwriting_features: Handwriting feature array  
            gait_features: Gait feature array
            voting_method: Voting method ('soft' or 'hard')
            calibration_context: Context for calibration (optional)
            
        Returns:
            Prediction result dictionary
        """
        context = calibration_context or {}
        
        return self._service.predict_ensemble(
            speech_features=speech_features,
            handwriting_features=handwriting_features,
            gait_features=gait_features,
            voting_method=voting_method,
            calibration_context=context
        )
    
    def predict_single_modality(
        self,
        modality: str,
        features: np.ndarray
    ) -> Dict[str, Any]:
        """
        Predict using single modality (backward-compatible API).
        
        Args:
            modality: Modality name
            features: Feature array
            
        Returns:
            Prediction result dictionary
        """
        return self._service.predict_single_modality(modality, features)
    
    def get_loaded_modalities(self) -> List[str]:
        """Get list of available modalities."""
        return ['speech', 'handwriting', 'gait']
    
    def is_model_loaded(self, modality: str) -> bool:
        """Check if model is loaded for modality."""
        return modality in self.get_loaded_modalities()
    
    def get_model_info(self) -> Dict:
        """Get model information."""
        return {
            'loaded_models': self.get_loaded_modalities(),
            'model_details': {}
        }


def get_model_manager() -> PredictionFacade:
    """
    Get model manager (backward-compatible factory).
    
    Returns existing architecture through facade pattern.
    External code can continue using get_model_manager()
    without knowing about the refactored architecture.
    """
    return PredictionFacade()

