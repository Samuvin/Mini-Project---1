"""
Video processing for gait analysis in Parkinson's Disease detection.
Extracts 10 gait features from walking videos.
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')


def extract_gait_features(video_path: str) -> Dict[str, float]:
    """
    Extract 10 gait features from walking video.
    
    Note: These are estimated features using basic motion detection.
    Professional gait analysis requires force plates or motion capture systems.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with 10 gait features
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        # Return default values if video can't be opened
        return get_default_features()
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30  # Default
    
    # Collect motion data
    motion_data = []
    prev_frame = None
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if prev_frame is not None:
            # Calculate frame difference (motion)
            frame_diff = cv2.absdiff(prev_frame, gray)
            _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
            
            # Calculate motion amount
            motion_amount = np.sum(thresh) / (255 * thresh.size)
            motion_data.append(motion_amount)
        
        prev_frame = gray
        
        # Limit processing to first 300 frames for performance
        if frame_count > 300:
            break
    
    cap.release()
    
    if len(motion_data) < 10:
        return get_default_features()
    
    # Extract features from motion data
    features = calculate_gait_features(motion_data, fps, frame_count)
    
    return features


def calculate_gait_features(motion_data: List[float], fps: float, frame_count: int) -> Dict[str, float]:
    """Calculate gait features from motion data."""
    motion_array = np.array(motion_data)
    
    features = {}
    
    # 1-2: Stride interval and variability
    # Detect peaks in motion (steps)
    steps = detect_steps(motion_array)
    
    if len(steps) > 1:
        step_intervals = np.diff(steps) / fps  # Convert to seconds
        features['stride_interval'] = float(np.mean(step_intervals) * 2)  # 2 steps = 1 stride
        features['stride_interval_std'] = float(np.std(step_intervals) * 2)
    else:
        features['stride_interval'] = 1.1
        features['stride_interval_std'] = 0.05
    
    # 3-4: Swing and stance time estimates
    # Swing time: low motion periods
    # Stance time: high motion periods
    motion_threshold = np.median(motion_array)
    swing_frames = np.sum(motion_array < motion_threshold)
    stance_frames = np.sum(motion_array >= motion_threshold)
    
    features['swing_time'] = float((swing_frames / fps) / max(len(steps), 1) * 0.4)
    features['stance_time'] = float((stance_frames / fps) / max(len(steps), 1) * 0.7)
    
    # 5: Double support time estimate
    features['double_support'] = float(features['stance_time'] * 0.35)
    
    # 6-7: Gait speed and cadence
    duration = frame_count / fps
    if duration > 0 and len(steps) > 0:
        cadence = (len(steps) * 60) / duration  # Steps per minute
        features['cadence'] = float(min(cadence, 150))
        
        # Estimate speed (assuming average step length)
        step_length = 0.6  # meters (average)
        features['gait_speed'] = float((len(steps) * step_length) / duration)
    else:
        features['cadence'] = 100.0
        features['gait_speed'] = 1.0
    
    # 8: Step length estimate
    features['step_length'] = float(features['gait_speed'] / (features['cadence'] / 60))
    
    # 9: Stride regularity (from motion consistency)
    regularity = 1.0 - (np.std(motion_array) / (np.mean(motion_array) + 1e-10))
    features['stride_regularity'] = float(np.clip(regularity, 0, 1))
    
    # 10: Gait asymmetry (from motion pattern)
    # Analyze first half vs second half of motion
    mid_point = len(motion_array) // 2
    first_half_mean = np.mean(motion_array[:mid_point])
    second_half_mean = np.mean(motion_array[mid_point:])
    asymmetry = abs(first_half_mean - second_half_mean) / (first_half_mean + second_half_mean + 1e-10)
    features['gait_asymmetry'] = float(np.clip(asymmetry, 0, 1))
    
    return features


def detect_steps(motion_data: np.ndarray, min_distance: int = 15) -> List[int]:
    """
    Detect steps from motion data by finding peaks.
    
    Args:
        motion_data: Array of motion amounts per frame
        min_distance: Minimum frames between steps
        
    Returns:
        List of frame indices where steps occur
    """
    # Smooth the data
    if len(motion_data) > 5:
        smoothed = np.convolve(motion_data, np.ones(5)/5, mode='same')
    else:
        smoothed = motion_data
    
    # Find peaks (local maxima)
    peaks = []
    for i in range(1, len(smoothed) - 1):
        if smoothed[i] > smoothed[i-1] and smoothed[i] > smoothed[i+1]:
            # Check if it's above threshold
            if smoothed[i] > np.mean(smoothed) * 0.8:
                peaks.append(i)
    
    # Filter peaks that are too close together
    if len(peaks) < 2:
        return peaks
    
    filtered_peaks = [peaks[0]]
    for peak in peaks[1:]:
        if peak - filtered_peaks[-1] >= min_distance:
            filtered_peaks.append(peak)
    
    return filtered_peaks


def get_default_features() -> Dict[str, float]:
    """Return default gait features when video processing fails."""
    return {
        'stride_interval': 1.1,
        'stride_interval_std': 0.05,
        'swing_time': 0.4,
        'stance_time': 0.7,
        'double_support': 0.25,
        'gait_speed': 1.0,
        'cadence': 100.0,
        'step_length': 0.6,
        'stride_regularity': 0.8,
        'gait_asymmetry': 0.1
    }


def get_feature_names() -> List[str]:
    """Get list of all 10 gait feature names in order."""
    return [
        'stride_interval',
        'stride_interval_std',
        'swing_time',
        'stance_time',
        'double_support',
        'gait_speed',
        'cadence',
        'step_length',
        'stride_regularity',
        'gait_asymmetry'
    ]


def features_dict_to_array(features: Dict[str, float]) -> List[float]:
    """Convert features dictionary to ordered array."""
    feature_names = get_feature_names()
    return [features[name] for name in feature_names]


if __name__ == "__main__":
    print("Gait Video Feature Extractor")
    print("10 features for Parkinson's Disease detection")
    print("\nFeature list:")
    for i, name in enumerate(get_feature_names(), 1):
        print(f"  {i}. {name}")
    print("\nNote: Features are estimated from video analysis.")
    print("For best accuracy, use professional gait analysis equipment.")

