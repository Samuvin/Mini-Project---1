#!/usr/bin/env python3
"""
Test multi-model ensemble predictions for all modalities.
"""

import json
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.model_manager import get_model_manager

def print_header(text):
    """Print a formatted header."""
    print(f"\n{'='*70}")
    print(f"{text}")
    print(f"{'='*70}")

def print_success(text):
    """Print success message."""
    print(f"✓ {text}")

def print_error(text):
    """Print error message."""
    print(f"✗ {text}")

def print_info(text):
    """Print info message."""
    print(f"ℹ {text}")

def test_single_modality(manager, modality, sample_type, features):
    """Test single modality prediction."""
    modality_map = {
        'speech': 'speech_features',
        'handwriting': 'handwriting_features',
        'gait': 'gait_features'
    }
    
    kwargs = {modality_map[modality]: features}
    
    try:
        result = manager.predict_ensemble(**kwargs)
        
        prediction_label = result['prediction_label']
        confidence = result['confidence']
        
        expected_label = 'Healthy' if sample_type == 'healthy' else 'Parkinson\'s Disease'
        is_correct = prediction_label == expected_label
        
        status = "✓" if is_correct else "✗"
        print(f"  {status} {modality.capitalize():12s} | {sample_type.capitalize():10s} → "
              f"{prediction_label:20s} ({confidence*100:5.2f}%)")
        
        return is_correct
    except Exception as e:
        print(f"  ✗ {modality.capitalize():12s} | {sample_type.capitalize():10s} → ERROR: {e}")
        return False

def test_multi_modality(manager, sample_type, speech, handwriting, gait, modalities):
    """Test multi-modality ensemble prediction."""
    kwargs = {}
    modality_str = ""
    
    if 'speech' in modalities:
        kwargs['speech_features'] = speech
        modality_str += "S"
    if 'handwriting' in modalities:
        kwargs['handwriting_features'] = handwriting
        modality_str += "H"
    if 'gait' in modalities:
        kwargs['gait_features'] = gait
        modality_str += "G"
    
    try:
        result = manager.predict_ensemble(**kwargs)
        
        prediction_label = result['prediction_label']
        confidence = result['confidence']
        
        expected_label = 'Healthy' if sample_type == 'healthy' else 'Parkinson\'s Disease'
        is_correct = prediction_label == expected_label
        
        status = "✓" if is_correct else "✗"
        print(f"  {status} {modality_str:12s} | {sample_type.capitalize():10s} → "
              f"{prediction_label:20s} ({confidence*100:5.2f}%)")
        
        if len(result['individual_predictions']) > 1:
            print(f"     Individual: ", end="")
            for mod, pred in result['individual_predictions'].items():
                print(f"{mod[0].upper()}:{pred['probabilities']['parkinsons']*100:.1f}% ", end="")
            print()
        
        return is_correct
    except Exception as e:
        print(f"  ✗ {modality_str:12s} | {sample_type.capitalize():10s} → ERROR: {e}")
        return False

def main():
    """Main test function."""
    print_header("MULTI-MODEL ENSEMBLE PREDICTION TEST")
    
    # Initialize model manager
    print_info("Initializing model manager...")
    manager = get_model_manager()
    
    loaded_models = manager.get_loaded_modalities()
    print_success(f"Loaded models: {', '.join(loaded_models)}")
    
    if not loaded_models:
        print_error("No models loaded! Train models first.")
        return
    
    # Load example data
    print_info("Loading example data...")
    examples_path = Path('webapp/static/examples/real_examples.json')
    
    try:
        with open(examples_path, 'r') as f:
            examples = json.load(f)
        print_success("Example data loaded successfully")
    except Exception as e:
        print_error(f"Failed to load examples: {e}")
        return
    
    # Extract features
    healthy_speech = examples['healthy']['speech_features']
    healthy_handwriting = examples['healthy']['handwriting_features']
    healthy_gait = examples['healthy']['gait_features']
    
    pd_speech = examples['parkinsons']['speech_features']
    pd_handwriting = examples['parkinsons']['handwriting_features']
    pd_gait = examples['parkinsons']['gait_features']
    
    test_results = []
    
    # Test 1: Single Modality Predictions
    print_header("TEST 1: Single Modality Predictions")
    print(f"  {'Modality':12s} | {'Sample':10s} → {'Result':20s} (Confidence)")
    print(f"  {'-'*66}")
    
    for modality in ['speech', 'handwriting', 'gait']:
        if manager.is_model_loaded(modality):
            if modality == 'speech':
                result_h = test_single_modality(manager, modality, 'healthy', healthy_speech)
                result_pd = test_single_modality(manager, modality, 'parkinsons', pd_speech)
            elif modality == 'handwriting':
                result_h = test_single_modality(manager, modality, 'healthy', healthy_handwriting)
                result_pd = test_single_modality(manager, modality, 'parkinsons', pd_handwriting)
            else:  # gait
                result_h = test_single_modality(manager, modality, 'healthy', healthy_gait)
                result_pd = test_single_modality(manager, modality, 'parkinsons', pd_gait)
            
            test_results.append((f'{modality}_healthy', result_h))
            test_results.append((f'{modality}_pd', result_pd))
    
    # Test 2: Two-Modality Combinations
    if len(loaded_models) >= 2:
        print_header("TEST 2: Two-Modality Ensemble Predictions")
        print(f"  {'Modalities':12s} | {'Sample':10s} → {'Result':20s} (Confidence)")
        print(f"  {'-'*66}")
        
        # Speech + Handwriting
        if manager.is_model_loaded('speech') and manager.is_model_loaded('handwriting'):
            result_h = test_multi_modality(manager, 'healthy', healthy_speech, healthy_handwriting, None, ['speech', 'handwriting'])
            result_pd = test_multi_modality(manager, 'parkinsons', pd_speech, pd_handwriting, None, ['speech', 'handwriting'])
            test_results.append(('speech_handwriting_healthy', result_h))
            test_results.append(('speech_handwriting_pd', result_pd))
        
        # Speech + Gait
        if manager.is_model_loaded('speech') and manager.is_model_loaded('gait'):
            result_h = test_multi_modality(manager, 'healthy', healthy_speech, None, healthy_gait, ['speech', 'gait'])
            result_pd = test_multi_modality(manager, 'parkinsons', pd_speech, None, pd_gait, ['speech', 'gait'])
            test_results.append(('speech_gait_healthy', result_h))
            test_results.append(('speech_gait_pd', result_pd))
        
        # Handwriting + Gait
        if manager.is_model_loaded('handwriting') and manager.is_model_loaded('gait'):
            result_h = test_multi_modality(manager, 'healthy', None, healthy_handwriting, healthy_gait, ['handwriting', 'gait'])
            result_pd = test_multi_modality(manager, 'parkinsons', None, pd_handwriting, pd_gait, ['handwriting', 'gait'])
            test_results.append(('handwriting_gait_healthy', result_h))
            test_results.append(('handwriting_gait_pd', result_pd))
    
    # Test 3: All Three Modalities
    if len(loaded_models) == 3:
        print_header("TEST 3: Three-Modality Ensemble Predictions")
        print(f"  {'Modalities':12s} | {'Sample':10s} → {'Result':20s} (Confidence)")
        print(f"  {'-'*66}")
        
        result_h = test_multi_modality(manager, 'healthy', healthy_speech, healthy_handwriting, healthy_gait, ['speech', 'handwriting', 'gait'])
        result_pd = test_multi_modality(manager, 'parkinsons', pd_speech, pd_handwriting, pd_gait, ['speech', 'handwriting', 'gait'])
        test_results.append(('all_modalities_healthy', result_h))
        test_results.append(('all_modalities_pd', result_pd))
    
    # Print summary
    print_header("TEST SUMMARY")
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    print(f"\nTests Passed: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed == total:
        print_success("All tests passed! ✓")
        print_info("The multi-model system correctly differentiates between healthy and PD samples.")
    else:
        print_error(f"{total - passed} test(s) failed!")
        print_info("Failed tests:")
        for name, result in test_results:
            if not result:
                print(f"  - {name}")
    
    print()

if __name__ == '__main__':
    main()

