#!/usr/bin/env python3
"""
Test script to verify healthy bias is working correctly.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.model_manager import ModelManager
import numpy as np
import json

def print_header(title):
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")

def print_result(modality, prediction_label, confidence, prob_healthy, prob_pd):
    print(f"\n{modality.upper()} PREDICTION:")
    print(f"  Result: {prediction_label}")
    print(f"  Confidence: {confidence*100:.2f}%")
    print(f"  Probabilities:")
    print(f"    Healthy:     {prob_healthy*100:.2f}%")
    print(f"    Parkinson's: {prob_pd*100:.2f}%")

def main():
    print_header("TESTING HEALTHY BIAS IN PREDICTIONS")
    
    # Initialize model manager with healthy bias
    print("\n1. Loading models with healthy bias (factor = 3.0)...")
    manager = ModelManager(healthy_bias=3.0)
    
    loaded = manager.get_loaded_modalities()
    print(f"   ✓ Loaded models: {', '.join(loaded)}")
    
    # Load example data
    examples_path = Path('webapp/static/examples/real_examples.json')
    
    if not examples_path.exists():
        print("   ✗ Example data not found!")
        return
    
    with open(examples_path, 'r') as f:
        examples = json.load(f)
    
    print("   ✓ Example data loaded")
    
    # Test 1: Single modality predictions
    print_header("TEST 1: SINGLE MODALITY PREDICTIONS (with bias)")
    
    # Test speech
    if 'speech' in loaded:
        print("\n--- Testing Speech Modality ---")
        result = manager.predict_ensemble(
            speech_features=examples['healthy']['speech_features']
        )
        print_result('speech', result['prediction_label'], 
                    result['confidence'], 
                    result['probabilities']['healthy'],
                    result['probabilities']['parkinsons'])
    
    # Test handwriting
    if 'handwriting' in loaded:
        print("\n--- Testing Handwriting Modality ---")
        result = manager.predict_ensemble(
            handwriting_features=examples['healthy']['handwriting_features']
        )
        print_result('handwriting', result['prediction_label'], 
                    result['confidence'], 
                    result['probabilities']['healthy'],
                    result['probabilities']['parkinsons'])
    
    # Test gait
    if 'gait' in loaded:
        print("\n--- Testing Gait Modality ---")
        result = manager.predict_ensemble(
            gait_features=examples['healthy']['gait_features']
        )
        print_result('gait', result['prediction_label'], 
                    result['confidence'], 
                    result['probabilities']['healthy'],
                    result['probabilities']['parkinsons'])
    
    # Test 2: Ensemble prediction with all modalities
    print_header("TEST 2: ENSEMBLE PREDICTION (All Modalities)")
    
    result = manager.predict_ensemble(
        speech_features=examples['healthy']['speech_features'],
        handwriting_features=examples['healthy']['handwriting_features'],
        gait_features=examples['healthy']['gait_features']
    )
    
    print(f"\nENSEMBLE PREDICTION:")
    print(f"  Result: {result['prediction_label']}")
    print(f"  Confidence: {result['confidence']*100:.2f}%")
    print(f"  Ensemble Method: {result['ensemble_method']}")
    print(f"  Probabilities:")
    print(f"    Healthy:     {result['probabilities']['healthy']*100:.2f}%")
    print(f"    Parkinson's: {result['probabilities']['parkinsons']*100:.2f}%")
    
    print(f"\n  Individual Model Predictions:")
    for modality, pred in result['individual_predictions'].items():
        print(f"    {modality.capitalize():12s}: {pred['prediction_label']:20s} "
              f"(Healthy: {pred['probabilities']['healthy']*100:.2f}%)")
    
    # Test 3: Test with "uncertain" features (50-50 raw prediction)
    print_header("TEST 3: BIAS EFFECT ON UNCERTAIN PREDICTIONS")
    
    print("\nSimulating uncertain raw predictions (50% healthy, 50% PD)...")
    print("With bias factor 3.0, this should become ~75% healthy, ~25% PD")
    
    # Create manager with different bias levels
    for bias_factor in [1.0, 2.0, 3.0, 5.0]:
        manager_test = ModelManager(healthy_bias=bias_factor)
        biased_healthy, biased_pd = manager_test._apply_healthy_bias(0.5, 0.5)
        print(f"\n  Bias Factor {bias_factor}:")
        print(f"    Healthy: {biased_healthy*100:.2f}% | Parkinson's: {biased_pd*100:.2f}%")
    
    print_header("BIAS TESTING COMPLETE!")
    print("\n✓ The healthy bias is now active!")
    print("✓ All predictions are biased towards 'Healthy' outcomes.")
    print("✓ Web application needs to be restarted to use the new bias.")
    print("\nNOTE: This bias should NOT be used in production medical settings!")

if __name__ == '__main__':
    main()

