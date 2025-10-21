#!/usr/bin/env python3
"""
Create realistic example files for demonstration using actual dataset samples.
Uses REAL healthy samples from the UCI Parkinson's dataset.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

def create_example_features():
    """
    Extract REAL healthy sample features from the UCI Parkinson's dataset
    and save them as JSON for the web interface to use.
    """
    
    print("="*70)
    print("CREATING REALISTIC EXAMPLES FROM ACTUAL DATASET")
    print("="*70)
    print()
    
    # Load the speech dataset
    speech_path = Path("data/raw/speech/parkinsons.csv")
    if not speech_path.exists():
        print("❌ Speech dataset not found. Run download_data.py first.")
        return
    
    speech_df = pd.read_csv(speech_path)
    
    # Get healthy and PD samples for speech
    speech_healthy = speech_df[speech_df['status'] == 0]
    speech_pd = speech_df[speech_df['status'] == 1]
    print(f"Speech: {len(speech_healthy)} healthy, {len(speech_pd)} PD samples")
    
    # Load handwriting dataset
    handwriting_path = Path("data/raw/handwriting/handwriting_features.csv")
    hw_healthy, hw_pd = [], []
    if handwriting_path.exists():
        hw_df = pd.read_csv(handwriting_path)
        hw_healthy = hw_df[hw_df['status'] == 0]
        hw_pd = hw_df[hw_df['status'] == 1]
        print(f"Handwriting: {len(hw_healthy)} healthy, {len(hw_pd)} PD samples")
    else:
        print("⚠ Handwriting dataset not found")
    
    # Load gait dataset
    gait_path = Path("data/raw/gait/gait_features.csv")
    gait_healthy, gait_pd = [], []
    if gait_path.exists():
        gait_df = pd.read_csv(gait_path)
        gait_healthy = gait_df[gait_df['status'] == 0]
        gait_pd = gait_df[gait_df['status'] == 1]
        print(f"Gait: {len(gait_healthy)} healthy, {len(gait_pd)} PD samples")
    else:
        print("⚠ Gait dataset not found")
    
    print()
    
    # Select diverse samples
    if len(speech_healthy) > 0 and len(speech_pd) > 0:
        # Get speech features (middle of distribution)
        healthy_speech = speech_healthy.iloc[len(speech_healthy)//2]
        pd_speech = speech_pd.iloc[len(speech_pd)//2]
        
        speech_features = [col for col in speech_df.columns if col not in ['name', 'status']]
        
        # Get handwriting features
        healthy_hw = []
        pd_hw = []
        if len(hw_healthy) > 0 and len(hw_pd) > 0:
            hw_sample_healthy = hw_healthy.iloc[len(hw_healthy)//2]
            hw_sample_pd = hw_pd.iloc[len(hw_pd)//2]
            hw_features = [col for col in hw_df.columns if col not in ['status', 'subject_id', 'id']]
            healthy_hw = hw_sample_healthy[hw_features].tolist()
            pd_hw = hw_sample_pd[hw_features].tolist()
        
        # Get gait features
        healthy_gait = []
        pd_gait = []
        if len(gait_healthy) > 0 and len(gait_pd) > 0:
            gait_sample_healthy = gait_healthy.iloc[len(gait_healthy)//2]
            gait_sample_pd = gait_pd.iloc[len(gait_pd)//2]
            gait_features = [col for col in gait_df.columns if col not in ['status', 'subject_id']]
            healthy_gait = gait_sample_healthy[gait_features].tolist()
            pd_gait = gait_sample_pd[gait_features].tolist()
        
        examples = {
            'healthy': {
                'name': 'Healthy Control Sample',
                'source': 'Real Patient Data',
                'speech_features': healthy_speech[speech_features].tolist(),
                'handwriting_features': healthy_hw,
                'gait_features': healthy_gait
            },
            'parkinsons': {
                'name': 'Parkinson\'s Disease Sample',
                'source': 'Real Patient Data',
                'speech_features': pd_speech[speech_features].tolist(),
                'handwriting_features': pd_hw,
                'gait_features': pd_gait
            }
        }
        
        # Save to JSON
        output_dir = Path("webapp/static/examples")
        output_dir.mkdir(exist_ok=True, parents=True)
        
        output_file = output_dir / "real_examples.json"
        with open(output_file, 'w') as f:
            json.dump(examples, f, indent=2)
        
        print("✅ Created realistic examples:")
        print(f"   Saved to: {output_file}")
        print()
        print("HEALTHY SAMPLE:")
        print(f"  Speech: {len(examples['healthy']['speech_features'])} features")
        print(f"  Handwriting: {len(examples['healthy']['handwriting_features'])} features")
        print(f"  Gait: {len(examples['healthy']['gait_features'])} features")
        print(f"  Total: {len(examples['healthy']['speech_features']) + len(examples['healthy']['handwriting_features']) + len(examples['healthy']['gait_features'])} features")
        print()
        print("PARKINSON'S SAMPLE:")
        print(f"  Speech: {len(examples['parkinsons']['speech_features'])} features")
        print(f"  Handwriting: {len(examples['parkinsons']['handwriting_features'])} features")
        print(f"  Gait: {len(examples['parkinsons']['gait_features'])} features")
        print(f"  Total: {len(examples['parkinsons']['speech_features']) + len(examples['parkinsons']['handwriting_features']) + len(examples['parkinsons']['gait_features'])} features")
        print()
        print("="*70)
        print("✅ ALL FEATURES FROM REAL PATIENTS - NO SYNTHETIC DATA!")
        print("="*70)
    else:
        print("❌ Not enough samples in dataset")

if __name__ == "__main__":
    create_example_features()

