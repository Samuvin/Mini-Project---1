"""Gait feature extraction for Parkinson's Disease detection.

This module processes REAL gait data from PhysioNet Gait in Parkinson's Disease Database.
NO synthetic data generation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


# Feature descriptions for gait analysis
GAIT_FEATURE_DESCRIPTIONS = {
    'stride_interval': 'Time between successive heel strikes (seconds)',
    'stride_interval_std': 'Standard deviation of stride intervals',
    'swing_time': 'Duration foot is off the ground (seconds)',
    'stance_time': 'Duration foot is on the ground (seconds)',
    'double_support': 'Time both feet on ground (seconds)',
    'gait_speed': 'Walking velocity (meters/second)',
    'cadence': 'Steps per minute',
    'step_length': 'Distance covered per step (meters)',
    'stride_regularity': 'Consistency of gait pattern (0-1)',
    'gait_asymmetry': 'Left-right gait differences (0-1)'
}


def get_feature_names() -> List[str]:
    """
    Get list of gait feature names.
    
    Returns:
        List of feature names
    """
    return list(GAIT_FEATURE_DESCRIPTIONS.keys())


def get_feature_descriptions() -> Dict[str, str]:
    """
    Get descriptions of all gait features.
    
    Returns:
        Dictionary mapping feature names to descriptions
    """
    return GAIT_FEATURE_DESCRIPTIONS.copy()


def validate_gait_data(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate gait dataset format.
    
    Args:
        df: DataFrame containing gait features
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check if DataFrame is empty
    if df.empty:
        return False, "DataFrame is empty"
    
    # Check for status column
    if 'status' not in df.columns:
        return False, "Missing 'status' column (required for labels)"
    
    # Check status values
    if not df['status'].isin([0, 1]).all():
        return False, "Status column must contain only 0 (healthy) or 1 (PD)"
    
    # Check for numeric features
    feature_cols = [col for col in df.columns if col != 'status']
    if len(feature_cols) == 0:
        return False, "No feature columns found"
    
    # Check if features are numeric
    non_numeric = df[feature_cols].select_dtypes(exclude=[np.number]).columns
    if len(non_numeric) > 0:
        return False, f"Non-numeric features found: {list(non_numeric)}"
    
    return True, "Valid"


def extract_features_from_stride_data(stride_intervals: np.ndarray) -> Dict[str, float]:
    """
    Extract gait features from stride interval time series.
    
    This function processes stride interval data from PhysioNet database
    and calculates relevant features.
    
    Args:
        stride_intervals: Array of stride interval times (seconds)
        
    Returns:
        Dictionary of extracted features
    """
    if len(stride_intervals) < 10:
        raise ValueError("Need at least 10 stride intervals for feature extraction")
    
    features = {
        'stride_interval': float(np.mean(stride_intervals)),
        'stride_interval_std': float(np.std(stride_intervals)),
        'stride_regularity': float(1.0 - (np.std(stride_intervals) / np.mean(stride_intervals))),
    }
    
    # Calculate cadence (steps per minute)
    mean_stride = np.mean(stride_intervals)
    if mean_stride > 0:
        features['cadence'] = float(120.0 / mean_stride)  # 2 steps per stride
    else:
        features['cadence'] = 0.0
    
    return features


def process_physionet_file(filepath: str) -> Dict[str, float]:
    """
    Process a PhysioNet gait database file and extract features.
    
    PhysioNet files contain stride interval data in text format.
    
    Args:
        filepath: Path to PhysioNet data file
        
    Returns:
        Dictionary of extracted features
    """
    # Read stride intervals from file
    try:
        # PhysioNet files typically have stride intervals as single column
        stride_data = np.loadtxt(filepath)
        
        # Extract features
        features = extract_features_from_stride_data(stride_data)
        
        return features
    except Exception as e:
        raise ValueError(f"Error processing PhysioNet file {filepath}: {e}")


def get_feature_statistics(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Calculate statistics for each gait feature.
    
    Args:
        df: DataFrame containing gait features
        
    Returns:
        Dictionary with statistics for each feature
    """
    feature_cols = [col for col in df.columns if col != 'status']
    
    stats = {}
    for col in feature_cols:
        stats[col] = {
            'mean': float(df[col].mean()),
            'std': float(df[col].std()),
            'min': float(df[col].min()),
            'max': float(df[col].max()),
            'median': float(df[col].median())
        }
    
    return stats


def typical_feature_ranges() -> Dict[str, Tuple[float, float]]:
    """
    Get typical ranges for gait features in healthy vs PD patients.
    
    Returns:
        Dictionary mapping features to (healthy_range, pd_range) tuples
    """
    return {
        'stride_interval': ((1.0, 1.2), (1.1, 1.4)),  # PD: longer, more variable
        'stride_interval_std': ((0.02, 0.05), (0.05, 0.15)),  # PD: higher variability
        'gait_speed': ((1.2, 1.5), (0.8, 1.2)),  # PD: slower
        'cadence': ((110, 120), (90, 110)),  # PD: lower cadence
    }


if __name__ == "__main__":
    # Print feature information
    print("Gait Features for Parkinson's Disease Detection")
    print("=" * 60)
    print("\nFeature Descriptions:")
    for name, desc in GAIT_FEATURE_DESCRIPTIONS.items():
        print(f"  {name}: {desc}")
    
    print("\n\nTypical Feature Ranges:")
    ranges = typical_feature_ranges()
    for name, (healthy, pd) in ranges.items():
        print(f"  {name}:")
        print(f"    Healthy: {healthy}")
        print(f"    Parkinson's: {pd}")
    
    print("\n" + "=" * 60)

