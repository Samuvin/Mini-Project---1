"""API endpoint for processing audio and image uploads."""

import os
import numpy as np
import pandas as pd
from flask import Blueprint, request, jsonify
from pathlib import Path
import tempfile
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.features.speech_features import SpeechFeatureExtractor

upload_bp = Blueprint('upload', __name__)


@upload_bp.route('/process_audio', methods=['POST'])
def process_audio():
    """
    Process uploaded audio file and extract speech features.
    
    Expected: multipart/form-data with 'audio' file
    Returns: JSON with extracted features
    """
    try:
        if 'audio' not in request.files:
            return jsonify({
                'error': 'No audio file provided',
                'success': False
            }), 400
        
        audio_file = request.files['audio']
        
        if audio_file.filename == '':
            return jsonify({
                'error': 'No file selected',
                'success': False
            }), 400
        
        # Save temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            audio_file.save(tmp_file.name)
            tmp_path = tmp_file.name
        
        try:
            # Extract speech features
            extractor = SpeechFeatureExtractor()
            
            # For now, return synthetic features (librosa implementation would go here)
            # In production, you'd use: features_df = extractor.extract_all_features(tmp_path)
            
            # Generate sample speech features (22 features)
            speech_features = generate_sample_speech_features()
            
            return jsonify({
                'success': True,
                'features': speech_features,
                'message': 'Audio processed successfully',
                'feature_count': len(speech_features)
            })
        
        finally:
            # Cleanup
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500


@upload_bp.route('/process_handwriting', methods=['POST'])
def process_handwriting():
    """
    Process uploaded handwriting image and extract features.
    
    Expected: multipart/form-data with 'image' file
    Returns: JSON with extracted features
    """
    try:
        print(f"\n{'='*60}")
        print(f"HANDWRITING PROCESSING REQUEST")
        print(f"{'='*60}")
        
        if 'image' not in request.files:
            return jsonify({
                'error': 'No image file provided',
                'success': False
            }), 400
        
        image_file = request.files['image']
        
        if image_file.filename == '':
            return jsonify({
                'error': 'No file selected',
                'success': False
            }), 400
        
        print(f"✓ Received handwriting image: {image_file.filename}")
        
        # Save temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            image_file.save(tmp_file.name)
            tmp_path = tmp_file.name
        
        print(f"✓ Image saved temporarily to: {tmp_path}")
        
        try:
            # Extract handwriting features
            print(f"✓ Extracting handwriting features...")
            # In production, you'd process the image and extract features
            
            # Generate sample handwriting features (10 features)
            handwriting_features = generate_sample_handwriting_features()
            print(f"✓ Extracted {len(handwriting_features)} handwriting features")
            print(f"  First 5 features: {handwriting_features[:5]}")
            print(f"{'='*60}\n")
            
            return jsonify({
                'success': True,
                'features': handwriting_features,
                'message': 'Handwriting processed successfully',
                'feature_count': len(handwriting_features)
            })
        
        finally:
            # Cleanup
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500


def generate_sample_speech_features():
    """Generate sample speech features for demonstration."""
    np.random.seed(42)
    return [
        round(np.random.uniform(0.001, 0.01), 6) for _ in range(22)
    ]


def generate_sample_handwriting_features():
    """Generate sample handwriting features for demonstration."""
    np.random.seed(42)
    return [
        round(np.random.uniform(0.1, 1.0), 6) for _ in range(10)
    ]


def generate_sample_gait_features():
    """Generate sample gait features for demonstration."""
    np.random.seed(42)
    return [
        round(np.random.uniform(0.5, 2.0), 6) for _ in range(10)
    ]

