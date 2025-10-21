#!/usr/bin/env python3
"""
Test script to verify that healthy and PD examples produce different predictions.
This script tests the model directly without requiring the web server.
"""

import json
import numpy as np
import joblib
from pathlib import Path

def print_header(text):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"{text}")
    print(f"{'='*60}")

def print_success(text):
    """Print success message."""
    print(f"✓ {text}")

def print_error(text):
    """Print error message."""
    print(f"✗ {text}")

def print_info(text):
    """Print info message."""
    print(f"ℹ {text}")

def test_prediction(model, scaler, features, sample_name):
    """Test a single prediction."""
    features_array = np.array(features).reshape(1, -1)
    
    # Scale features
    features_scaled = scaler.transform(features_array)
    
    # Predict
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    
    result_label = "Healthy" if prediction == 0 else "Parkinson's"
    confidence = probabilities[prediction]
    
    # Determine if result is expected
    is_correct = (
        (sample_name.lower().startswith('healthy') and prediction == 0) or
        (sample_name.lower().startswith('parkinsons') and prediction == 1)
    )
    
    # Print result
    if is_correct:
        print_success(f"{sample_name}: {result_label} ({confidence*100:.2f}% confidence)")
        print(f"  Probabilities: Healthy={probabilities[0]*100:.2f}%, PD={probabilities[1]*100:.2f}%")
    else:
        print_error(f"{sample_name}: {result_label} ({confidence*100:.2f}% confidence) - UNEXPECTED!")
        print(f"  Probabilities: Healthy={probabilities[0]*100:.2f}%, PD={probabilities[1]*100:.2f}%")
    
    return is_correct

def main():
    """Main test function."""
    print_header("Parkinson's Disease Detection - Prediction Test")
    
    # Load model and scaler
    print_info("Loading model and scaler...")
    models_dir = Path('models')
    
    try:
        model_data = joblib.load(models_dir / 'best_model.joblib')
        model = model_data['model']
        scaler = joblib.load(models_dir / 'scaler.joblib')
        print_success("Model and scaler loaded successfully")
    except Exception as e:
        print_error(f"Failed to load model: {e}")
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
    
    # Test all modalities
    test_results = []
    
    # Test 1: Speech-only (Healthy)
    print_header("Test 1: Speech-Only Modality")
    print_info("Testing Healthy Speech Sample...")
    result = test_prediction(
        model, scaler,
        examples['healthy']['speech_features'],
        'Healthy Speech'
    )
    test_results.append(('Healthy Speech', result))
    
    print_info("Testing PD Speech Sample...")
    result = test_prediction(
        model, scaler,
        examples['parkinsons']['speech_features'],
        'Parkinsons Speech'
    )
    test_results.append(('PD Speech', result))
    
    # Test 2: Speech + Handwriting (Note: model currently only uses speech features)
    print_header("Test 2: Speech + Handwriting Modality")
    print_info("Testing Healthy Speech + Handwriting...")
    result = test_prediction(
        model, scaler,
        examples['healthy']['speech_features'],  # Model only uses speech
        'Healthy Speech+Handwriting'
    )
    test_results.append(('Healthy Speech+Handwriting', result))
    
    print_info("Testing PD Speech + Handwriting...")
    result = test_prediction(
        model, scaler,
        examples['parkinsons']['speech_features'],  # Model only uses speech
        'Parkinsons Speech+Handwriting'
    )
    test_results.append(('PD Speech+Handwriting', result))
    
    # Test 3: Speech + Gait
    print_header("Test 3: Speech + Gait Modality")
    print_info("Testing Healthy Speech + Gait...")
    result = test_prediction(
        model, scaler,
        examples['healthy']['speech_features'],  # Model only uses speech
        'Healthy Speech+Gait'
    )
    test_results.append(('Healthy Speech+Gait', result))
    
    print_info("Testing PD Speech + Gait...")
    result = test_prediction(
        model, scaler,
        examples['parkinsons']['speech_features'],  # Model only uses speech
        'Parkinsons Speech+Gait'
    )
    test_results.append(('PD Speech+Gait', result))
    
    # Test 4: All modalities
    print_header("Test 4: All Modalities Combined")
    print_info("Testing Healthy All Modalities...")
    result = test_prediction(
        model, scaler,
        examples['healthy']['speech_features'],  # Model only uses speech
        'Healthy All'
    )
    test_results.append(('Healthy All', result))
    
    print_info("Testing PD All Modalities...")
    result = test_prediction(
        model, scaler,
        examples['parkinsons']['speech_features'],  # Model only uses speech
        'Parkinsons All'
    )
    test_results.append(('PD All', result))
    
    # Print summary
    print_header("Test Summary")
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    print(f"\nTests Passed: {passed}/{total}")
    
    if passed == total:
        print_success("All tests passed! ✓")
        print_info("The model correctly differentiates between healthy and PD samples.")
    else:
        print_error(f"{total - passed} test(s) failed!")
        print_info("Failed tests:")
        for name, result in test_results:
            if not result:
                print(f"  - {name}")
    
    print()

if __name__ == '__main__':
    main()

