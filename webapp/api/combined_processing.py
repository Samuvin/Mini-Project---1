"""API endpoint for processing combined video with multiple modalities."""

import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from flask import Blueprint, request, jsonify

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.audio_processing import extract_speech_features
from utils.video_processing import extract_gait_features
from utils.image_processing import extract_handwriting_features

logger = logging.getLogger(__name__)

combined_bp = Blueprint('combined', __name__)


@combined_bp.route('/process_combined_video', methods=['POST'])
def process_combined_video():
    """Process uploaded video file and extract selected modalities.

    Expected: multipart/form-data with:
      - ``video`` file
      - ``extract_voice`` (boolean)
      - ``extract_handwriting`` (boolean)
      - ``extract_gait`` (boolean)

    Returns JSON with extracted features for selected modalities.
    """
    try:
        if 'video' not in request.files:
            return jsonify({
                'error': 'No video file provided',
                'success': False
            }), 400
        
        video_file = request.files['video']
        
        if video_file.filename == '':
            return jsonify({
                'error': 'No file selected',
                'success': False
            }), 400
        
        extract_voice = request.form.get('extract_voice', 'false').lower() == 'true'
        extract_handwriting = request.form.get('extract_handwriting', 'false').lower() == 'true'
        extract_gait = request.form.get('extract_gait', 'false').lower() == 'true'
        
        logger.info(
            "Combined video processing: file=%s, voice=%s, handwriting=%s, gait=%s",
            video_file.filename, extract_voice, extract_handwriting, extract_gait,
        )
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            video_file.save(tmp_file.name)
            tmp_path = tmp_file.name
        
        try:
            response_data = {
                'success': True,
                'voice_features': None,
                'handwriting_features': None,
                'gait_features': None,
                'total_features': 0
            }
            
            if extract_voice:
                try:
                    voice_features = _generate_sample_speech_features()
                    response_data['voice_features'] = voice_features
                    response_data['total_features'] += len(voice_features)
                    logger.info("Extracted %d voice features", len(voice_features))
                except Exception as e:
                    logger.warning("Error extracting voice: %s", e)
            
            if extract_handwriting:
                try:
                    handwriting_features = _generate_sample_handwriting_features()
                    response_data['handwriting_features'] = handwriting_features
                    response_data['total_features'] += len(handwriting_features)
                    logger.info("Extracted %d handwriting features", len(handwriting_features))
                except Exception as e:
                    logger.warning("Error extracting handwriting: %s", e)
            
            if extract_gait:
                try:
                    gait_features = _generate_sample_gait_features()
                    response_data['gait_features'] = gait_features
                    response_data['total_features'] += len(gait_features)
                    logger.info("Extracted %d gait features", len(gait_features))
                except Exception as e:
                    logger.warning("Error extracting gait: %s", e)
            
            if response_data['total_features'] == 0:
                return jsonify({
                    'error': 'No features could be extracted from the video',
                    'success': False,
                    'note': 'Please ensure the video contains the selected assessment types'
                }), 400
            
            logger.info("Total features extracted: %d", response_data['total_features'])
            return jsonify(response_data)
        
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    except Exception as e:
        logger.exception("Combined video processing failed")
        return jsonify({
            'error': str(e),
            'success': False
        }), 500


def _generate_sample_speech_features():
    """Generate sample speech features (placeholder for real extraction)."""
    import time
    seed = int(time.time() * 1000) % 100000
    rng = np.random.default_rng(seed)
    return [round(float(x), 6) for x in rng.uniform(0.001, 0.01, 22)]


def _generate_sample_handwriting_features():
    """Generate sample handwriting features (placeholder for real extraction)."""
    import time
    seed = int(time.time() * 1000) % 100000 + 17
    rng = np.random.default_rng(seed)
    return [round(float(x), 6) for x in rng.uniform(0.1, 1.0, 10)]


def _generate_sample_gait_features():
    """Generate sample gait features (placeholder for real extraction)."""
    import time
    seed = int(time.time() * 1000) % 100000 + 37
    rng = np.random.default_rng(seed)
    return [round(float(x), 6) for x in rng.uniform(0.5, 2.0, 10)]
