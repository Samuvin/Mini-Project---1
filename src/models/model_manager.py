"""
Multi-Model Manager for Parkinson's Disease Prediction.
Handles loading and ensemble predictions from speech, handwriting, and gait models.
"""

import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages multiple PD prediction models and provides ensemble predictions.
    """
    
    def __init__(self, models_dir: str = 'models'):
        """
        Initialize the model manager.
        
        Args:
            models_dir: Directory containing trained models
        """
        self.models_dir = Path(models_dir)
        self.models = {}
        self.scalers = {}
        self.model_info = {}
        
        # Load all available models
        self._load_models()
    
    def _load_models(self):
        """Load all available models (speech, handwriting, gait)."""
        modalities = ['speech', 'handwriting', 'gait']
        
        for modality in modalities:
            model_path = self.models_dir / f'{modality}_model.joblib'
            scaler_path = self.models_dir / f'{modality}_scaler.joblib'
            
            # For speech, also check for best_model.joblib (legacy)
            if modality == 'speech' and not model_path.exists():
                legacy_path = self.models_dir / 'best_model.joblib'
                if legacy_path.exists():
                    model_path = legacy_path
            
            if model_path.exists() and scaler_path.exists():
                try:
                    model_data = joblib.load(model_path)
                    scaler = joblib.load(scaler_path)
                    
                    # Handle different model formats
                    if isinstance(model_data, dict):
                        self.models[modality] = model_data['model']
                        self.model_info[modality] = {
                            'name': model_data.get('model_name', 'Unknown'),
                            'n_features': model_data.get('n_features', 0),
                            'feature_names': model_data.get('feature_names', [])
                        }
                    else:
                        self.models[modality] = model_data
                        self.model_info[modality] = {
                            'name': type(model_data).__name__,
                            'n_features': 0,
                            'feature_names': []
                        }
                    
                    self.scalers[modality] = scaler
                    logger.info(f"✓ Loaded {modality} model: {self.model_info[modality]['name']}")
                    
                except Exception as e:
                    logger.warning(f"Failed to load {modality} model: {e}")
            else:
                logger.info(f"ℹ {modality.capitalize()} model not found (optional)")
    
    def is_model_loaded(self, modality: str) -> bool:
        """Check if a specific modality model is loaded."""
        return modality in self.models
    
    def get_loaded_modalities(self) -> List[str]:
        """Get list of loaded modalities."""
        return list(self.models.keys())
    
    def predict_single_modality(
        self, 
        modality: str, 
        features: np.ndarray
    ) -> Dict:
        """
        Make prediction using a single modality model.
        
        Args:
            modality: One of 'speech', 'handwriting', 'gait'
            features: Feature array for the modality
            
        Returns:
            Dictionary with prediction results
        """
        if modality not in self.models:
            raise ValueError(f"Model for {modality} not loaded")
        
        # Ensure features are 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Scale features
        features_scaled = self.scalers[modality].transform(features)
        
        # Make prediction
        model = self.models[modality]
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        return {
            'modality': modality,
            'prediction': int(prediction),
            'prediction_label': 'Parkinson\'s Disease' if prediction == 1 else 'Healthy',
            'confidence': float(probabilities[prediction]),
            'probabilities': {
                'healthy': float(probabilities[0]),
                'parkinsons': float(probabilities[1])
            }
        }
    
    def predict_ensemble(
        self,
        speech_features: Optional[np.ndarray] = None,
        handwriting_features: Optional[np.ndarray] = None,
        gait_features: Optional[np.ndarray] = None,
        voting_method: str = 'soft'
    ) -> Dict:
        """
        Make ensemble prediction using available modalities.
        
        Args:
            speech_features: Speech features (22 features)
            handwriting_features: Handwriting features (10 features)
            gait_features: Gait features (10 features)
            voting_method: 'soft' (probability averaging) or 'hard' (majority vote)
            
        Returns:
            Dictionary with ensemble prediction results
        """
        # Collect available modalities
        available_features = {}
        if speech_features is not None and self.is_model_loaded('speech'):
            available_features['speech'] = np.array(speech_features)
        if handwriting_features is not None and self.is_model_loaded('handwriting'):
            available_features['handwriting'] = np.array(handwriting_features)
        if gait_features is not None and self.is_model_loaded('gait'):
            available_features['gait'] = np.array(gait_features)
        
        if not available_features:
            raise ValueError("No modalities available for prediction")
        
        # Get predictions from each modality
        modality_results = {}
        for modality, features in available_features.items():
            result = self.predict_single_modality(modality, features)
            modality_results[modality] = result
        
        # If only one modality, return its result directly
        if len(modality_results) == 1:
            single_result = list(modality_results.values())[0]
            return {
                'success': True,
                'prediction': single_result['prediction'],
                'prediction_label': single_result['prediction_label'],
                'confidence': single_result['confidence'],
                'probabilities': single_result['probabilities'],
                'modalities_used': list(modality_results.keys()),
                'individual_predictions': modality_results,
                'ensemble_method': 'single_modality'
            }
        
        # Ensemble voting
        if voting_method == 'soft':
            # Average probabilities across modalities
            avg_prob_healthy = np.mean([
                r['probabilities']['healthy'] for r in modality_results.values()
            ])
            avg_prob_pd = np.mean([
                r['probabilities']['parkinsons'] for r in modality_results.values()
            ])
            
            ensemble_prediction = 1 if avg_prob_pd > avg_prob_healthy else 0
            ensemble_confidence = max(avg_prob_healthy, avg_prob_pd)
            
        else:  # hard voting
            # Majority vote
            predictions = [r['prediction'] for r in modality_results.values()]
            ensemble_prediction = 1 if sum(predictions) > len(predictions) / 2 else 0
            
            # Average confidence of models that agree with ensemble
            agreeing_confidences = [
                r['confidence'] for r in modality_results.values()
                if r['prediction'] == ensemble_prediction
            ]
            ensemble_confidence = np.mean(agreeing_confidences) if agreeing_confidences else 0.5
            
            avg_prob_healthy = 1 - ensemble_confidence if ensemble_prediction == 1 else ensemble_confidence
            avg_prob_pd = ensemble_confidence if ensemble_prediction == 1 else 1 - ensemble_confidence
        
        return {
            'success': True,
            'prediction': int(ensemble_prediction),
            'prediction_label': 'Parkinson\'s Disease' if ensemble_prediction == 1 else 'Healthy',
            'confidence': float(ensemble_confidence),
            'probabilities': {
                'healthy': float(avg_prob_healthy),
                'parkinsons': float(avg_prob_pd)
            },
            'modalities_used': list(modality_results.keys()),
            'individual_predictions': modality_results,
            'ensemble_method': voting_method
        }
    
    def get_model_info(self) -> Dict:
        """Get information about loaded models."""
        return {
            'loaded_models': list(self.models.keys()),
            'model_details': self.model_info
        }


# Global model manager instance (singleton)
_model_manager = None


def get_model_manager() -> ModelManager:
    """Get or create the global model manager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager


def reload_models():
    """Reload all models (useful after retraining)."""
    global _model_manager
    _model_manager = None
    return get_model_manager()

