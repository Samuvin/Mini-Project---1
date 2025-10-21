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

predict_bp = Blueprint('predict', __name__)

# Global variables to store loaded model and scaler
_model = None
_scaler = None
_model_loaded = False


def load_model(model_path: str = None):
    """Load the trained model and scaler."""
    global _model, _scaler, _model_loaded
    
    if _model_loaded:
        return _model, _scaler
    
    models_dir = get_models_dir()
    
    # Try to load SVM model first, then logistic regression
    model_files = [
        'best_model.joblib',
        'svm_model.joblib',
        'logistic_regression_model.joblib'
    ]
    
    if model_path:
        model_files.insert(0, model_path)
    
    for model_file in model_files:
        try:
            filepath = models_dir / model_file
            if filepath.exists():
                model_data = joblib.load(filepath)
                _model = model_data['model']
                _model_loaded = True
                print(f"Model loaded successfully from {filepath}")
                
                # Try to load the scaler
                scaler_path = models_dir / 'scaler.joblib'
                if scaler_path.exists():
                    _scaler = joblib.load(scaler_path)
                    print(f"Scaler loaded successfully from {scaler_path}")
                else:
                    print("Warning: No scaler found. Predictions may be inaccurate.")
                    _scaler = None
                
                return _model, _scaler
        except Exception as e:
            print(f"Error loading model from {model_file}: {e}")
            continue
    
    print("Warning: No trained model found. Please train a model first.")
    return None, None


@predict_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    model, scaler = load_model()
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None
    })


@predict_bp.route('/predict', methods=['POST'])
def predict():
    """
    Make a prediction based on input features.
    
    Expected JSON format (Option 1 - Combined):
    {
        "features": [all feature values concatenated]
    }
    
    Expected JSON format (Option 2 - Separate modalities):
    {
        "speech_features": [22 speech values],
        "handwriting_features": [10 handwriting values],
        "gait_features": [10 gait values]
    }
    """
    try:
        model, scaler = load_model()
        
        if model is None:
            return jsonify({
                'error': 'Model not loaded. Please train a model first.',
                'success': False
            }), 500
        
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'No data provided',
                'success': False
            }), 400
        
        # Extract features - support flexible modality inputs
        if 'features' in data:
            # Combined features format
            features = np.array(data['features']).reshape(1, -1)
        else:
            # Build features from available modalities with padding
            # Model expects 42 features: 22 speech + 10 handwriting + 10 gait
            
            # Initialize with neutral values (zeros work well with scaled data)
            speech_features = np.zeros(22)
            handwriting_features = np.zeros(10)
            gait_features = np.zeros(10)
            
            modalities_provided = []
            
            if 'speech_features' in data and data['speech_features']:
                speech = np.array(data['speech_features'])
                if len(speech) == 22:
                    speech_features = speech
                    modalities_provided.append('speech')
                    print(f"  ✓ Speech features: {len(speech)}")
                else:
                    print(f"  ⚠ Warning: Expected 22 speech features, got {len(speech)}")
            
            if 'handwriting_features' in data and data['handwriting_features']:
                handwriting = np.array(data['handwriting_features'])
                if len(handwriting) == 10:
                    handwriting_features = handwriting
                    modalities_provided.append('handwriting')
                    print(f"  ✓ Handwriting features: {len(handwriting)}")
                else:
                    print(f"  ⚠ Warning: Expected 10 handwriting features, got {len(handwriting)}")
            
            if 'gait_features' in data and data['gait_features']:
                gait = np.array(data['gait_features'])
                if len(gait) == 10:
                    gait_features = gait
                    modalities_provided.append('gait')
                    print(f"  ✓ Gait features: {len(gait)}")
                else:
                    print(f"  ⚠ Warning: Expected 10 gait features, got {len(gait)}")
            
            if not modalities_provided:
                return jsonify({
                    'error': 'No features provided. Please provide at least one modality (speech, handwriting, or gait).',
                    'success': False
                }), 400
            
            # Concatenate all features (with padding for missing modalities)
            features = np.concatenate([speech_features, handwriting_features, gait_features]).reshape(1, -1)
            print(f"  Modalities used: {', '.join(modalities_provided)}")
            print(f"  Total features (with padding): {features.shape[1]}")
        
        # Scale features if scaler is available
        print(f"\n{'='*60}")
        print(f"PREDICTION REQUEST RECEIVED")
        print(f"{'='*60}")
        print(f"Input features shape: {features.shape}")
        print(f"Input features (first 5): {features[0][:5]}")
        
        if scaler is not None:
            print("✓ Scaling features using StandardScaler...")
            features = scaler.transform(features)
            print(f"Scaled features (first 5): {features[0][:5]}")
        else:
            print("⚠ Warning: Making prediction without scaling - results may be inaccurate")
        
        # Make prediction
        print(f"✓ Running SVM model prediction...")
        prediction = model.predict(features)[0]
        prediction_proba = model.predict_proba(features)[0]
        
        # Debug logging
        print(f"✓ Prediction complete!")
        print(f"  Result: {'Parkinson\'s Disease' if prediction == 1 else 'Healthy'}")
        print(f"  Confidence: {prediction_proba[prediction]*100:.2f}%")
        print(f"  Probabilities: Healthy={prediction_proba[0]*100:.2f}%, PD={prediction_proba[1]*100:.2f}%")
        print(f"{'='*60}\n")
        
        # Prepare response
        result = {
            'success': True,
            'prediction': int(prediction),
            'prediction_label': 'Parkinson\'s Disease' if prediction == 1 else 'Healthy',
            'confidence': float(prediction_proba[prediction]),
            'probabilities': {
                'healthy': float(prediction_proba[0]),
                'parkinsons': float(prediction_proba[1])
            },
            'debug_info': {
                'features_count': int(features.shape[1]),
                'model_type': type(model).__name__
            }
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500


@predict_bp.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Make predictions for multiple samples.
    
    Expected JSON format:
    {
        "features": [[list of features for sample 1], [list of features for sample 2], ...]
    }
    """
    try:
        model, scaler = load_model()
        
        if model is None:
            return jsonify({
                'error': 'Model not loaded. Please train a model first.',
                'success': False
            }), 500
        
        data = request.get_json()
        
        if not data or 'features' not in data:
            return jsonify({
                'error': 'No features provided',
                'success': False
            }), 400
        
        features = np.array(data['features'])
        
        # Scale features if scaler is available
        if scaler is not None:
            features = scaler.transform(features)
        
        # Make predictions
        predictions = model.predict(features)
        predictions_proba = model.predict_proba(features)
        
        # Prepare results
        results = []
        for i, (pred, proba) in enumerate(zip(predictions, predictions_proba)):
            results.append({
                'sample_id': i,
                'prediction': int(pred),
                'prediction_label': 'Parkinson\'s Disease' if pred == 1 else 'Healthy',
                'confidence': float(proba[pred]),
                'probabilities': {
                    'healthy': float(proba[0]),
                    'parkinsons': float(proba[1])
                }
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
    """Get information about the loaded model."""
    try:
        model, scaler = load_model()
        
        if model is None:
            return jsonify({
                'error': 'Model not loaded',
                'success': False
            }), 500
        
        # Get model information
        info = {
            'success': True,
            'model_type': type(model).__name__,
            'kernel': getattr(model, 'kernel', 'N/A'),
            'n_features': model.n_features_in_ if hasattr(model, 'n_features_in_') else 'Unknown',
            'scaler_loaded': scaler is not None
        }
        
        return jsonify(info)
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

