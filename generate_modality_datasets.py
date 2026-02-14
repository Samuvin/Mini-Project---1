#!/usr/bin/env python3
"""
Generate synthetic training datasets for handwriting and gait-based PD prediction.
Based on published research characteristics and realistic feature distributions.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json

np.random.seed(42)

def generate_handwriting_features(n_samples, is_pd=False):
    """
    Generate 10 handwriting features based on PD research:
    1. Mean Pressure (0-1, PD: lower)
    2. Pressure Variation (0-1, PD: higher)
    3. Mean Velocity (mm/s, PD: slower)
    4. Velocity Variation (0-1, PD: higher)
    5. Mean Acceleration (mm/s², PD: more variable)
    6. Pen-up Time Ratio (0-1, PD: higher)
    7. Mean Stroke Length (mm, PD: shorter)
    8. Writing Tempo (strokes/s, PD: slower)
    9. Tremor Power (frequency domain, PD: higher)
    10. Fluency Score (0-1, PD: lower)
    """
    features = []
    
    for _ in range(n_samples):
        if is_pd:
            # PD characteristics: reduced pressure, slower, more tremor
            mean_pressure = np.random.normal(0.45, 0.08)
            pressure_var = np.random.normal(0.18, 0.04)
            mean_velocity = np.random.normal(2.1, 0.5)
            velocity_var = np.random.normal(0.72, 0.12)
            mean_accel = np.random.normal(0.85, 0.25)
            penup_time = np.random.normal(0.38, 0.08)
            stroke_length = np.random.normal(6.5, 1.5)
            tempo = np.random.normal(2.0, 0.5)
            tremor = np.random.normal(0.68, 0.15)
            fluency = np.random.normal(0.42, 0.12)
        else:
            # Healthy characteristics: normal pressure, faster, less tremor
            mean_pressure = np.random.normal(0.72, 0.10)
            pressure_var = np.random.normal(0.11, 0.03)
            mean_velocity = np.random.normal(3.2, 0.6)
            velocity_var = np.random.normal(0.45, 0.10)
            mean_accel = np.random.normal(1.45, 0.35)
            penup_time = np.random.normal(0.22, 0.06)
            stroke_length = np.random.normal(10.5, 2.0)
            tempo = np.random.normal(3.2, 0.6)
            tremor = np.random.normal(0.28, 0.10)
            fluency = np.random.normal(0.78, 0.10)
        
        # Clip values to realistic ranges
        feature = [
            np.clip(mean_pressure, 0.1, 1.0),
            np.clip(pressure_var, 0.01, 0.5),
            np.clip(mean_velocity, 0.5, 8.0),
            np.clip(velocity_var, 0.1, 1.5),
            np.clip(mean_accel, 0.2, 3.0),
            np.clip(penup_time, 0.05, 0.8),
            np.clip(stroke_length, 2.0, 20.0),
            np.clip(tempo, 0.5, 5.0),
            np.clip(tremor, 0.01, 1.0),
            np.clip(fluency, 0.1, 1.0)
        ]
        features.append(feature)
    
    return np.array(features)


def generate_gait_features(n_samples, is_pd=False):
    """
    Generate 10 gait features based on PD research:
    1. Stride Interval (s, PD: more variable)
    2. Stride Variability (CV, PD: higher)
    3. Swing Time (s, PD: reduced)
    4. Stance Time (s, PD: increased)
    5. Double Support Time (s, PD: increased)
    6. Gait Speed (m/s, PD: slower)
    7. Cadence (steps/min, PD: reduced)
    8. Step Length (m, PD: shorter)
    9. Stride Regularity (0-1, PD: lower)
    10. Gait Asymmetry (0-1, PD: higher)
    """
    features = []
    
    for _ in range(n_samples):
        if is_pd:
            # PD characteristics: slower, more variable, less regular
            stride_interval = np.random.normal(1.15, 0.12)
            stride_var = np.random.normal(0.08, 0.025)
            swing_time = np.random.normal(0.35, 0.05)
            stance_time = np.random.normal(0.72, 0.08)
            double_support = np.random.normal(0.32, 0.08)
            gait_speed = np.random.normal(0.85, 0.20)
            cadence = np.random.normal(98, 15)
            step_length = np.random.normal(0.52, 0.12)
            regularity = np.random.normal(0.68, 0.15)
            asymmetry = np.random.normal(0.18, 0.08)
        else:
            # Healthy characteristics: normal speed, regular, symmetric
            stride_interval = np.random.normal(1.05, 0.05)
            stride_var = np.random.normal(0.03, 0.01)
            swing_time = np.random.normal(0.42, 0.04)
            stance_time = np.random.normal(0.62, 0.05)
            double_support = np.random.normal(0.18, 0.04)
            gait_speed = np.random.normal(1.25, 0.15)
            cadence = np.random.normal(115, 10)
            step_length = np.random.normal(0.72, 0.10)
            regularity = np.random.normal(0.92, 0.05)
            asymmetry = np.random.normal(0.06, 0.03)
        
        # Clip values to realistic ranges
        feature = [
            np.clip(stride_interval, 0.8, 1.5),
            np.clip(stride_var, 0.01, 0.2),
            np.clip(swing_time, 0.2, 0.6),
            np.clip(stance_time, 0.4, 0.9),
            np.clip(double_support, 0.1, 0.5),
            np.clip(gait_speed, 0.3, 2.0),
            np.clip(cadence, 60, 140),
            np.clip(step_length, 0.3, 1.0),
            np.clip(regularity, 0.3, 1.0),
            np.clip(asymmetry, 0.0, 0.4)
        ]
        features.append(feature)
    
    return np.array(features)


def main():
    print("="*60)
    print("GENERATING SYNTHETIC TRAINING DATA")
    print("="*60)
    
    # Create directories
    handwriting_dir = Path('data/raw/handwriting')
    gait_dir = Path('data/raw/gait')
    handwriting_dir.mkdir(parents=True, exist_ok=True)
    gait_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate datasets (balanced classes)
    n_samples_per_class = 150  # 300 total per modality
    
    print("\n1. Generating Handwriting Dataset...")
    handwriting_healthy = generate_handwriting_features(n_samples_per_class, is_pd=False)
    handwriting_pd = generate_handwriting_features(n_samples_per_class, is_pd=True)
    
    handwriting_features = np.vstack([handwriting_healthy, handwriting_pd])
    handwriting_labels = np.hstack([
        np.zeros(n_samples_per_class),
        np.ones(n_samples_per_class)
    ])
    
    # Create DataFrame
    feature_names = [
        'mean_pressure', 'pressure_variation', 'mean_velocity', 'velocity_variation',
        'mean_acceleration', 'penup_time_ratio', 'mean_stroke_length', 'writing_tempo',
        'tremor_power', 'fluency_score'
    ]
    
    handwriting_df = pd.DataFrame(handwriting_features, columns=feature_names)
    handwriting_df['status'] = handwriting_labels.astype(int)
    handwriting_df['sample_id'] = [f'hw_sample_{i}' for i in range(len(handwriting_df))]
    
    # Reorder columns
    cols = ['sample_id'] + feature_names + ['status']
    handwriting_df = handwriting_df[cols]
    
    # Save
    handwriting_path = handwriting_dir / 'handwriting_data.csv'
    handwriting_df.to_csv(handwriting_path, index=False)
    print(f"   ✓ Saved {len(handwriting_df)} samples to {handwriting_path}")
    print(f"   - Healthy: {(handwriting_df['status']==0).sum()}")
    print(f"   - PD: {(handwriting_df['status']==1).sum()}")
    
    print("\n2. Generating Gait Dataset...")
    gait_healthy = generate_gait_features(n_samples_per_class, is_pd=False)
    gait_pd = generate_gait_features(n_samples_per_class, is_pd=True)
    
    gait_features = np.vstack([gait_healthy, gait_pd])
    gait_labels = np.hstack([
        np.zeros(n_samples_per_class),
        np.ones(n_samples_per_class)
    ])
    
    # Create DataFrame
    feature_names = [
        'stride_interval', 'stride_variability', 'swing_time', 'stance_time',
        'double_support_time', 'gait_speed', 'cadence', 'step_length',
        'stride_regularity', 'gait_asymmetry'
    ]
    
    gait_df = pd.DataFrame(gait_features, columns=feature_names)
    gait_df['status'] = gait_labels.astype(int)
    gait_df['sample_id'] = [f'gait_sample_{i}' for i in range(len(gait_df))]
    
    # Reorder columns
    cols = ['sample_id'] + feature_names + ['status']
    gait_df = gait_df[cols]
    
    # Save
    gait_path = gait_dir / 'gait_data.csv'
    gait_df.to_csv(gait_path, index=False)
    print(f"   ✓ Saved {len(gait_df)} samples to {gait_path}")
    print(f"   - Healthy: {(gait_df['status']==0).sum()}")
    print(f"   - PD: {(gait_df['status']==1).sum()}")
    
    # Print statistics
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    
    print("\nHandwriting Features (Healthy vs PD):")
    hw_cols = ['mean_pressure', 'mean_velocity', 'tremor_power', 'fluency_score']
    for col in hw_cols:
        healthy_mean = handwriting_df[handwriting_df['status']==0][col].mean()
        pd_mean = handwriting_df[handwriting_df['status']==1][col].mean()
        print(f"  {col:20s}: Healthy={healthy_mean:.3f}, PD={pd_mean:.3f}")
    
    print("\nGait Features (Healthy vs PD):")
    gait_cols = ['gait_speed', 'cadence', 'stride_regularity', 'gait_asymmetry']
    for col in gait_cols:
        healthy_mean = gait_df[gait_df['status']==0][col].mean()
        pd_mean = gait_df[gait_df['status']==1][col].mean()
        print(f"  {col:20s}: Healthy={healthy_mean:.3f}, PD={pd_mean:.3f}")
    
    print("\n" + "="*60)
    print("✓ Dataset generation complete!")
    print("="*60)


if __name__ == '__main__':
    main()

