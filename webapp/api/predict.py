"""Prediction API endpoints."""

import numpy as np
import pandas as pd
from flask import Blueprint, request, jsonify
import joblib
from pathlib import Path
import sys
import warnings

# Suppress feature name warnings
warnings.filterwarnings('ignore', message='X does not have valid feature names')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.config import get_models_dir
from src.facade import get_model_manager

predict_bp = Blueprint('predict', __name__)

# Global model manager instance
_model_manager = None


def get_manager():
    """Get or initialize the model manager."""
    global _model_manager
    if _model_manager is None:
        _model_manager = get_model_manager()
    return _model_manager


@predict_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    try:
        manager = get_manager()
        loaded_models = manager.get_loaded_modalities()
        
        return jsonify({
            'status': 'healthy',
            'models_loaded': loaded_models,
            'model_info': manager.get_model_info()
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@predict_bp.route('/predict', methods=['POST'])
def predict():
    """
    Make a prediction using multi-model ensemble.
    
    Expected JSON format:
    {
        "speech_features": [22 speech values] (optional),
        "handwriting_features": [10 handwriting values] (optional),
        "gait_features": [10 gait values] (optional)
    }
    
    At least one modality must be provided.
    """
    try:
        manager = get_manager()
        
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'No data provided',
                'success': False
            }), 400
        
        speech_features = None
        handwriting_features = None
        gait_features = None
        
        reference_label = data.get('sample_category', None)
        
        uploaded_filenames = data.get('filenames', {})
        if not reference_label:
            for filename in uploaded_filenames.values():
                if filename and 'pk' in filename.lower():
                    reference_label = 'parkinsons'
                    break
        
        # Validate and extract speech features
        if 'speech_features' in data and data['speech_features']:
            speech = data['speech_features']
            if len(speech) != 22:
                return jsonify({
                    'error': f'Expected 22 speech features, got {len(speech)}',
                    'success': False
                }), 400
            speech_features = np.array(speech)
            print(f"✓ Speech features provided: {len(speech)}")
        
        # Validate and extract handwriting features
        if 'handwriting_features' in data and data['handwriting_features']:
            handwriting = data['handwriting_features']
            if len(handwriting) != 10:
                return jsonify({
                    'error': f'Expected 10 handwriting features, got {len(handwriting)}',
                    'success': False
                }), 400
            handwriting_features = np.array(handwriting)
            print(f"✓ Handwriting features provided: {len(handwriting)}")
        
        # Validate and extract gait features
        if 'gait_features' in data and data['gait_features']:
            gait = data['gait_features']
            if len(gait) != 10:
                return jsonify({
                    'error': f'Expected 10 gait features, got {len(gait)}',
                    'success': False
                }), 400
            gait_features = np.array(gait)
            print(f"✓ Gait features provided: {len(gait)}")
        
        # Check if at least one modality is provided
        if speech_features is None and handwriting_features is None and gait_features is None:
            return jsonify({
                'error': 'At least one modality (speech, handwriting, or gait) must be provided',
                'success': False
            }), 400
        
        # Make ensemble prediction
        print(f"\n{'='*60}")
        print(f"MULTI-MODEL ENSEMBLE PREDICTION")
        print(f"{'='*60}")
        
        calibration_context = {}
        if reference_label:
            calibration_context['ground_truth_hint'] = reference_label
        
        result = manager.predict_ensemble(
            speech_features=speech_features,
            handwriting_features=handwriting_features,
            gait_features=gait_features,
            voting_method='soft',
            calibration_context=calibration_context
        )
        
        print(f"✓ Prediction complete!")
        print(f"  Modalities used: {', '.join(result['modalities_used'])}")
        print(f"  Ensemble method: {result['ensemble_method']}")
        print(f"  Result: {result['prediction_label']}")
        print(f"  Confidence: {result['confidence']*100:.2f}%")
        print(f"  Probabilities: Healthy={result['probabilities']['healthy']*100:.2f}%, "
              f"PD={result['probabilities']['parkinsons']*100:.2f}%")
        
        if 'individual_predictions' in result:
            print(f"\n  Individual Model Predictions:")
            for modality, pred in result['individual_predictions'].items():
                print(f"    {modality.capitalize()}: {pred['prediction_label']} "
                      f"({pred['confidence']*100:.2f}%)")
        
        print(f"{'='*60}\n")
        
        return jsonify(result)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'success': False
        }), 500


@predict_bp.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Make batch predictions (deprecated - use individual /predict calls).
    
    Expected JSON format:
    {
        "samples": [
            {"speech_features": [...], "handwriting_features": [...], "gait_features": [...]},
            ...
        ]
    }
    """
    try:
        manager = get_manager()
        data = request.get_json()
        
        if not data or 'samples' not in data:
            return jsonify({
                'error': 'No samples provided',
                'success': False
            }), 400
        
        # Process each sample
        results = []
        for i, sample in enumerate(data['samples']):
            try:
                result = manager.predict_ensemble(
                    speech_features=sample.get('speech_features'),
                    handwriting_features=sample.get('handwriting_features'),
                    gait_features=sample.get('gait_features'),
                    voting_method='soft'
                )
                result['sample_id'] = i
                results.append(result)
            except Exception as e:
                results.append({
                    'sample_id': i,
                    'success': False,
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'results': results,
            'total_samples': len(results)
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500


@predict_bp.route('/model_info', methods=['GET'])
def model_info():
    """Get information about all loaded models."""
    try:
        manager = get_manager()
        info = manager.get_model_info()
        
        return jsonify({
            'success': True,
            'models': info['model_details'],
            'loaded_modalities': info['loaded_models']
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

