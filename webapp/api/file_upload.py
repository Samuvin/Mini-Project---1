"""
File upload API endpoints for audio, images, and video.
Handles file uploads and extracts features automatically.
"""

from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
import os
import sys
from pathlib import Path
import tempfile
import traceback

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.audio_processing import extract_speech_features, features_dict_to_array as audio_to_array
from utils.image_processing import extract_handwriting_features, features_dict_to_array as image_to_array
from utils.video_processing import extract_gait_features, features_dict_to_array as video_to_array

upload_bp = Blueprint('upload', __name__)

# Allowed file extensions
ALLOWED_AUDIO = {'wav', 'mp3', 'ogg', 'm4a', 'flac'}
ALLOWED_IMAGE = {'jpg', 'jpeg', 'png', 'bmp', 'tiff'}
ALLOWED_VIDEO = {'mp4', 'avi', 'mov', 'mkv'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB


def allowed_file(filename, allowed_extensions):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


@upload_bp.route('/audio', methods=['POST'])
def upload_audio():
    """
    Upload audio file and extract 22 speech features.
    
    Returns:
        JSON with extracted features array
    """
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file provided'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        # Validate file type
        if not allowed_file(file.filename, ALLOWED_AUDIO):
            return jsonify({
                'success': False,
                'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_AUDIO)}'
            }), 400
        
        # Save to temporary file
        filename = secure_filename(file.filename)
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name
        
        try:
            # Extract features
            print(f"Extracting features from: {tmp_path}")
            features_dict = extract_speech_features(tmp_path)
            features_array = audio_to_array(features_dict)
            
            # Clean up
            os.unlink(tmp_path)
            
            # Check if example features were used (first value is 119.992)
            is_example = abs(features_array[0] - 119.992) < 0.001
            
            return jsonify({
                'success': True,
                'features': features_array,
                'feature_count': len(features_array),
                'feature_names': list(features_dict.keys()),
                'modality': 'speech',
                'message': 'Audio features extracted successfully',
                'note': 'Note: Browser audio format not fully supported. Using reference features for demonstration.' if is_example else 'Features extracted from your audio'
            })
        
        except Exception as e:
            # Clean up on error
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise e
    
    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Error processing audio file: {str(e)}'
        }), 500


@upload_bp.route('/handwriting', methods=['POST'])
def upload_handwriting():
    """
    Upload handwriting image and extract 10 features.
    
    Returns:
        JSON with extracted features array
    """
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file provided'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        # Validate file type
        if not allowed_file(file.filename, ALLOWED_IMAGE):
            return jsonify({
                'success': False,
                'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_IMAGE)}'
            }), 400
        
        # Save to temporary file
        filename = secure_filename(file.filename)
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name
        
        try:
            # Extract features
            features_dict = extract_handwriting_features(tmp_path)
            features_array = image_to_array(features_dict)
            
            # Clean up
            os.unlink(tmp_path)
            
            return jsonify({
                'success': True,
                'features': features_array,
                'feature_count': len(features_array),
                'feature_names': list(features_dict.keys()),
                'modality': 'handwriting',
                'message': 'Handwriting features extracted successfully',
                'note': 'Optional features that enhance prediction accuracy when combined with speech'
            })
        
        except Exception as e:
            # Clean up on error
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise e
    
    except Exception as e:
        print(f"Error processing handwriting: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Error processing handwriting image: {str(e)}'
        }), 500


@upload_bp.route('/gait', methods=['POST'])
def upload_gait():
    """
    Upload walking video and extract 10 gait features.
    
    Returns:
        JSON with extracted features array
    """
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file provided'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        # Validate file type
        if not allowed_file(file.filename, ALLOWED_VIDEO):
            return jsonify({
                'success': False,
                'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_VIDEO)}'
            }), 400
        
        # Save to temporary file
        filename = secure_filename(file.filename)
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name
        
        try:
            # Extract features
            features_dict = extract_gait_features(tmp_path)
            features_array = video_to_array(features_dict)
            
            # Clean up
            os.unlink(tmp_path)
            
            return jsonify({
                'success': True,
                'features': features_array,
                'feature_count': len(features_array),
                'feature_names': list(features_dict.keys()),
                'modality': 'gait',
                'message': 'Gait features extracted successfully',
                'note': 'Optional features that enhance prediction accuracy when combined with speech'
            })
        
        except Exception as e:
            # Clean up on error
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise e
    
    except Exception as e:
        print(f"Error processing gait video: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Error processing gait video: {str(e)}'
        }), 500


@upload_bp.route('/test', methods=['GET'])
def test_upload():
    """Test endpoint to verify upload API is working."""
    return jsonify({
        'success': True,
        'message': 'Upload API is working',
        'endpoints': {
            'audio': '/api/upload/audio',
            'handwriting': '/api/upload/handwriting',
            'gait': '/api/upload/gait'
        }
    })

