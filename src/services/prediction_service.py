"""
Prediction service implementing business logic.
Follows Dependency Inversion Principle - depends on abstractions, not concretions.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Any
from src.core.interfaces import (
    IPredictionService, 
    IModelLoader, 
    ICalibrator
)
from src.core.entities import (
    PredictionResult, 
    EnsemblePredictionResult, 
    FeatureVector
)

logger = logging.getLogger(__name__)


class MultiModalPredictionService(IPredictionService):
    """
    Service for multi-modal PD predictions.
    
    Dependencies are injected (DIP), making it testable and flexible.
    """
    
    def __init__(
        self, 
        model_loader: IModelLoader,
        calibrator: ICalibrator
    ):
        """
        Initialize with dependencies.
        
        Args:
            model_loader: Service for loading models
            calibrator: Service for calibrating confidence
        """
        self._model_loader = model_loader
        self._calibrator = calibrator
        self._available_modalities = ['speech', 'handwriting', 'gait']
    
    def predict_single_modality(
        self, 
        modality: str, 
        features: np.ndarray
    ) -> Dict[str, Any]:
        """
        Predict using single modality.
        
        Args:
            modality: One of speech/handwriting/gait
            features: Input features
            
        Returns:
            Prediction result dictionary
        """
        if modality not in self._available_modalities:
            raise ValueError(f"Unknown modality: {modality}")
        
        model = self._model_loader.load_model(modality)
        scaler = self._model_loader.load_scaler(modality)
        
        if not model or not scaler:
            raise RuntimeError(f"Model or scaler not available for: {modality}")
        
        feature_vec = FeatureVector(modality=modality, features=features)
        scaled_features = scaler.transform(feature_vec.features)
        
        prediction = model.predict(scaled_features)
        probabilities = model.predict_proba(scaled_features)
        
        result = PredictionResult(
            prediction=prediction,
            prediction_label='Parkinson\'s Disease' if prediction == 1 else 'Healthy',
            confidence=float(probabilities[prediction]),
            probabilities={
                'healthy': float(probabilities[0]),
                'parkinsons': float(probabilities[1])
            },
            modality=modality
        )
        
        return result.to_dict()
    
    def predict_ensemble(
        self,
        speech_features: Optional[np.ndarray] = None,
        handwriting_features: Optional[np.ndarray] = None,
        gait_features: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Predict using ensemble of available modalities.
        
        Implements soft voting with calibration.
        """
        features_map = {
            'speech': speech_features,
            'handwriting': handwriting_features,
            'gait': gait_features
        }
        
        available = {k: v for k, v in features_map.items() if v is not None}
        
        if not available:
            raise ValueError("At least one modality required for prediction")
        
        individual_results = {}
        for modality, features in available.items():
            try:
                result = self.predict_single_modality(modality, features)
                individual_results[modality] = PredictionResult(**result)
            except Exception as e:
                logger.error(f"Prediction failed for {modality}: {e}")
        
        if not individual_results:
            raise RuntimeError("All modality predictions failed")
        
        ensemble_result = self._compute_ensemble(
            individual_results, 
            list(available.values()),
            kwargs.get('voting_method', 'soft'),
            kwargs.get('calibration_context', {})
        )
        
        return ensemble_result.to_dict()
    
    def _compute_ensemble(
        self,
        individual_results: Dict[str, PredictionResult],
        feature_vectors: List,
        voting_method: str,
        calibration_context: Dict
    ) -> EnsemblePredictionResult:
        """Compute ensemble prediction from individual results."""
        if len(individual_results) == 1:
            single_result = list(individual_results.values())[0]
            return self._create_single_modality_result(
                single_result, 
                individual_results,
                feature_vectors,
                calibration_context
            )
        
        if voting_method == 'soft':
            return self._soft_voting_ensemble(
                individual_results,
                feature_vectors,
                calibration_context
            )
        else:
            return self._hard_voting_ensemble(
                individual_results,
                feature_vectors,
                calibration_context
            )
    
    def _soft_voting_ensemble(
        self,
        results: Dict[str, PredictionResult],
        features: List,
        context: Dict
    ) -> EnsemblePredictionResult:
        """Soft voting by averaging probabilities."""
        avg_healthy = np.mean([r.probabilities['healthy'] for r in results.values()])
        avg_pd = np.mean([r.probabilities['parkinsons'] for r in results.values()])
        
        prediction = 1 if avg_pd > avg_healthy else 0
        
        calibrated = self._calibrator.calibrate(
            prediction=prediction,
            probabilities={'healthy': avg_healthy, 'parkinsons': avg_pd},
            features=features,
            context=context
        )
        
        return EnsemblePredictionResult(
            prediction=calibrated['prediction'],
            prediction_label=calibrated['prediction_label'],
            confidence=calibrated['confidence'],
            probabilities=calibrated['probabilities'],
            modalities_used=list(results.keys()),
            individual_predictions=results,
            ensemble_method='soft'
        )
    
    def _hard_voting_ensemble(
        self,
        results: Dict[str, PredictionResult],
        features: List,
        context: Dict
    ) -> EnsemblePredictionResult:
        """Hard voting by majority."""
        predictions = [r.prediction for r in results.values()]
        prediction = 1 if sum(predictions) > len(predictions) / 2 else 0
        
        agreeing = [r.confidence for r in results.values() if r.prediction == prediction]
        confidence = float(np.mean(agreeing)) if agreeing else 0.5
        
        calibrated = self._calibrator.calibrate(
            prediction=prediction,
            probabilities={
                'healthy': 1 - confidence if prediction == 1 else confidence,
                'parkinsons': confidence if prediction == 1 else 1 - confidence
            },
            features=features,
            context=context
        )
        
        return EnsemblePredictionResult(
            prediction=calibrated['prediction'],
            prediction_label=calibrated['prediction_label'],
            confidence=calibrated['confidence'],
            probabilities=calibrated['probabilities'],
            modalities_used=list(results.keys()),
            individual_predictions=results,
            ensemble_method='hard'
        )
    
    def _create_single_modality_result(
        self,
        single_result: PredictionResult,
        all_results: Dict,
        features: List,
        context: Dict
    ) -> EnsemblePredictionResult:
        """Create ensemble result for single modality."""
        calibrated = self._calibrator.calibrate(
            prediction=single_result.prediction,
            probabilities=single_result.probabilities,
            features=features,
            context=context
        )
        
        return EnsemblePredictionResult(
            prediction=calibrated['prediction'],
            prediction_label=calibrated['prediction_label'],
            confidence=calibrated['confidence'],
            probabilities=calibrated['probabilities'],
            modalities_used=list(all_results.keys()),
            individual_predictions=all_results,
            ensemble_method='single_modality'
        )

