"""Handwriting feature extraction for Parkinson's Disease detection.

This module processes REAL handwriting data from PaHaW or NewHandPD datasets.
NO synthetic data generation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


# Feature descriptions for handwriting analysis
HANDWRITING_FEATURE_DESCRIPTIONS = {
    'mean_pressure': 'Average pen pressure during writing',
    'std_pressure': 'Standard deviation of pen pressure',
    'mean_velocity': 'Average writing velocity',
    'std_velocity': 'Standard deviation of velocity',
    'mean_acceleration': 'Average pen acceleration',
    'pen_up_time': 'Total time pen lifted from paper',
    'stroke_length': 'Average stroke length',
    'writing_tempo': 'Overall writing speed/tempo',
    'tremor_frequency': 'Frequency of tremor in writing motion',
    'fluency_score': 'Writing fluency measure (0-1)'
}


def get_feature_names() -> List[str]:
    """
    Get list of handwriting feature names.
    
    Returns:
        List of feature names
    """
    return list(HANDWRITING_FEATURE_DESCRIPTIONS.keys())


def get_feature_descriptions() -> Dict[str, str]:
    """
    Get descriptions of all handwriting features.
    
    Returns:
        Dictionary mapping feature names to descriptions
    """
    return HANDWRITING_FEATURE_DESCRIPTIONS.copy()


def validate_handwriting_data(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate handwriting dataset format.
    
    Args:
        df: DataFrame containing handwriting features
        
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


def extract_features_from_raw(raw_data: pd.DataFrame) -> pd.DataFrame:
    """
    Extract handwriting features from raw pen trajectory data.
    
    This function processes time-series pen data (x, y, pressure, timestamp)
    and extracts relevant features for Parkinson's detection.
    
    Args:
        raw_data: DataFrame with columns [x, y, pressure, timestamp, status]
        
    Returns:
        DataFrame with extracted features
    """
    # This is a placeholder for actual feature extraction
    # Real implementation would depend on the raw data format
    raise NotImplementedError(
        "Feature extraction from raw pen data is not yet implemented.\n"
        "Please provide pre-extracted features in CSV format.\n"
        "See DATASETS.md for expected format."
    )


def get_feature_statistics(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Calculate statistics for each feature.
    
    Args:
        df: DataFrame containing handwriting features
        
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


if __name__ == "__main__":
    # Print feature information
    print("Handwriting Features for Parkinson's Disease Detection")
    print("=" * 60)
    print("\nFeature Descriptions:")
    for name, desc in HANDWRITING_FEATURE_DESCRIPTIONS.items():
        print(f"  {name}: {desc}")
    print("\n" + "=" * 60)

