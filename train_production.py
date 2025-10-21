"""Production training pipeline with nested CV and ensemble models for Parkinson's Disease Detection."""

import sys
from pathlib import Path
import json
import time
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data.data_loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.models.logistic_regression import LogisticRegressionModel
from src.models.svm_model import SVMModel
from src.models.random_forest import RandomForestModel
from src.models.gradient_boosting import XGBoostModel, LightGBMModel
from src.models.ensemble import EnsembleModel, create_ensemble_from_trained_models
from src.evaluation.metrics import ModelEvaluator
from src.evaluation.stability import StabilityAnalyzer
from src.utils.config import Config, get_models_dir

import pandas as pd
import numpy as np


def main():
    """Main production training pipeline."""
    print("\n" + "="*80)
    print("PARKINSON'S DISEASE DETECTION - PRODUCTION TRAINING PIPELINE")
    print("="*80 + "\n")
    
    start_time = time.time()
    
    # Initialize configuration
    config = Config()
    val_config = config.config.get('validation', {})
    use_nested_cv = val_config.get('use_nested_cv', False)
    
    print(f"Configuration loaded successfully!")
    print(f"Nested CV: {'Enabled' if use_nested_cv else 'Disabled'}")
    print(f"Random seed: {config.random_state}\n")
    
    # Step 1: Load Data with Telemonitoring Dataset
    print("Step 1: Loading multimodal datasets with enhanced speech data...")
    print("-" * 80)
    loader = DataLoader(config)
    
    try:
        # Try to load combined speech data (includes telemonitoring)
        X_speech = loader.load_combined_speech_data()
        X, y = X_speech
        print(f"\n✓ Enhanced speech data loaded successfully!")
        print(f"  Total: {X.shape[0]} samples, {X.shape[1]} features\n")
    except Exception as e:
        print(f"\n⚠ Could not load combined speech data: {e}")
        print("Falling back to standard speech data...")
        X, y = loader.load_speech_data()
        print(f"\n✓ Standard speech data loaded: {X.shape[0]} samples, {X.shape[1]} features\n")
    
    # Step 2: Preprocess Data with Augmentation
    print("\nStep 2: Preprocessing data with augmentation...")
    print("-" * 80)
    preprocessor = DataPreprocessor(config)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.preprocess_pipeline(
        X, y, 
        remove_outliers=False,
        balance_classes=True,
        apply_augmentation=True,
        save=True
    )
    
    print(f"Final dataset sizes:")
    print(f"  Training: {X_train.shape[0]} samples")
    print(f"  Validation: {X_val.shape[0]} samples")
    print(f"  Test: {X_test.shape[0]} samples\n")
    
    # Initialize models
    models = {}
    trained_models = {}
    test_results = {}
    
    # Step 3: Train Logistic Regression Baseline
    print("\n\nStep 3: Training Logistic Regression Baseline...")
    print("-" * 80)
    lr_model = LogisticRegressionModel(config)
    lr_history = lr_model.train(X_train, y_train, X_val, y_val)
    lr_test_metrics = lr_model.evaluate(X_test, y_test)
    lr_model.save_model('logistic_regression_model.joblib')
    
    models['Logistic Regression'] = lr_model
    trained_models['LR'] = lr_model
    test_results['Logistic Regression'] = lr_test_metrics
    
    # Step 4: Train SVM
    print("\n\nStep 4: Training SVM with Kernel Optimization...")
    print("-" * 80)
    svm_model = SVMModel(config)
    svm_history = svm_model.train(X_train, y_train, X_val, y_val, search_method='grid')
    svm_model.print_kernel_comparison()
    svm_test_metrics = svm_model.evaluate(X_test, y_test)
    svm_model.save_model('svm_model.joblib')
    
    models['SVM'] = svm_model
    trained_models['SVM'] = svm_model
    test_results['SVM'] = svm_test_metrics
    
    # Step 5: Train Random Forest
    print("\n\nStep 5: Training Random Forest...")
    print("-" * 80)
    rf_model = RandomForestModel(config)
    rf_history = rf_model.train(X_train, y_train, X_val, y_val, search_method='grid')
    rf_test_metrics = rf_model.evaluate(X_test, y_test)
    rf_model.save_model('random_forest_model.joblib')
    
    models['Random Forest'] = rf_model
    trained_models['RF'] = rf_model
    test_results['Random Forest'] = rf_test_metrics
    
    # Step 6: Train XGBoost
    print("\n\nStep 6: Training XGBoost...")
    print("-" * 80)
    xgb_model = XGBoostModel(config)
    xgb_history = xgb_model.train(X_train, y_train, X_val, y_val, search_method='random')
    xgb_test_metrics = xgb_model.evaluate(X_test, y_test)
    xgb_model.save_model('xgboost_model.joblib')
    
    models['XGBoost'] = xgb_model
    trained_models['XGBoost'] = xgb_model
    test_results['XGBoost'] = xgb_test_metrics
    
    # Step 7: Train LightGBM
    print("\n\nStep 7: Training LightGBM...")
    print("-" * 80)
    lgb_model = LightGBMModel(config)
    lgb_history = lgb_model.train(X_train, y_train, X_val, y_val, search_method='random')
    lgb_test_metrics = lgb_model.evaluate(X_test, y_test)
    lgb_model.save_model('lightgbm_model.joblib')
    
    models['LightGBM'] = lgb_model
    trained_models['LightGBM'] = lgb_model
    test_results['LightGBM'] = lgb_test_metrics
    
    # Step 8: Create Voting Ensemble
    print("\n\nStep 8: Creating Voting Ensemble...")
    print("-" * 80)
    voting_ensemble = create_ensemble_from_trained_models(
        trained_models,
        ensemble_type='voting',
        config=config
    )
    
    # Train ensemble
    ensemble_estimators = [(name, model.model) for name, model in trained_models.items()]
    voting_history = voting_ensemble.train_voting_ensemble(
        ensemble_estimators, X_train, y_train, X_val, y_val
    )
    voting_test_metrics = voting_ensemble.evaluate(X_test, y_test, 'Voting Ensemble')
    voting_ensemble.save_model('voting_ensemble_model.joblib')
    
    models['Voting Ensemble'] = voting_ensemble
    test_results['Voting Ensemble'] = voting_test_metrics
    
    # Step 9: Create Stacking Ensemble
    print("\n\nStep 9: Creating Stacking Ensemble...")
    print("-" * 80)
    stacking_ensemble = EnsembleModel(config)
    stacking_history = stacking_ensemble.train_stacking_ensemble(
        ensemble_estimators, X_train, y_train, X_val, y_val
    )
    stacking_test_metrics = stacking_ensemble.evaluate(X_test, y_test, 'Stacking Ensemble')
    stacking_ensemble.save_model('stacking_ensemble_model.joblib')
    
    models['Stacking Ensemble'] = stacking_ensemble
    test_results['Stacking Ensemble'] = stacking_test_metrics
    
    # Step 10: Model Comparison
    print("\n\nStep 10: Comprehensive Model Comparison...")
    print("-" * 80)
    
    evaluator = ModelEvaluator(config)
    comparison = evaluator.compare_models(
        test_results,
        save_path=get_models_dir() / 'comprehensive_model_comparison.png'
    )
    
    print("\n" + "="*80)
    print("Model Comparison Results (Test Set)")
    print("="*80)
    print(comparison.to_string())
    print()
    
    # Step 11: Stability Analysis
    print("\n\nStep 11: Stability Analysis...")
    print("-" * 80)
    
    stability_analyzer = StabilityAnalyzer(config)
    
    # Collect CV scores from training
    model_cv_scores = {}
    for name, model in models.items():
        if hasattr(model, 'training_history') and 'best_score' in model.training_history:
            # For models with grid search, we have CV scores
            if 'cv_results' in model.training_history:
                cv_results = model.training_history['cv_results']
                if 'mean_test_score' in cv_results:
                    # Get top 5 mean scores to analyze stability
                    top_scores = sorted(cv_results['mean_test_score'], reverse=True)[:5]
                    model_cv_scores[name] = np.array(top_scores)
    
    if model_cv_scores:
        # Create stability summary
        stability_summary = stability_analyzer.create_stability_summary(
            model_cv_scores,
            save_path=get_models_dir() / 'stability_summary.csv'
        )
        
        print("\nStability Summary:")
        print(stability_summary.to_string(index=False))
        
        # Statistical comparison
        if len(model_cv_scores) >= 2:
            comparison_df = stability_analyzer.compare_models_statistically(model_cv_scores)
    
    # Step 12: Select and Save Best Models
    print("\n\nStep 12: Selecting Best Models...")
    print("-" * 80)
    
    # Find best model by accuracy
    best_accuracy_model = max(test_results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\nBest Model (Accuracy): {best_accuracy_model[0]}")
    print(f"  Accuracy: {best_accuracy_model[1]['accuracy']:.4f}")
    
    # Find best model by F1-score
    best_f1_model = max(test_results.items(), key=lambda x: x[1]['f1_score'])
    print(f"\nBest Model (F1-Score): {best_f1_model[0]}")
    print(f"  F1-Score: {best_f1_model[1]['f1_score']:.4f}")
    
    # Find best model by ROC-AUC
    best_roc_model = max(test_results.items(), key=lambda x: x[1]['roc_auc'])
    print(f"\nBest Model (ROC-AUC): {best_roc_model[0]}")
    print(f"  ROC-AUC: {best_roc_model[1]['roc_auc']:.4f}")
    
    # Save best model
    best_model_name = best_roc_model[0]
    best_model = models[best_model_name]
    best_model.save_model('best_model.joblib')
    
    # Step 13: Save Training Metadata
    print("\n\nStep 13: Saving Training Metadata...")
    print("-" * 80)
    
    training_time = time.time() - start_time
    
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'training_time_seconds': training_time,
        'dataset_info': {
            'n_train_samples': int(X_train.shape[0]),
            'n_val_samples': int(X_val.shape[0]),
            'n_test_samples': int(X_test.shape[0]),
            'n_features': int(X_train.shape[1])
        },
        'best_models': {
            'by_accuracy': {
                'name': best_accuracy_model[0],
                'accuracy': float(best_accuracy_model[1]['accuracy'])
            },
            'by_f1_score': {
                'name': best_f1_model[0],
                'f1_score': float(best_f1_model[1]['f1_score'])
            },
            'by_roc_auc': {
                'name': best_roc_model[0],
                'roc_auc': float(best_roc_model[1]['roc_auc'])
            }
        },
        'all_models_test_results': {
            name: {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                   for k, v in metrics.items()}
            for name, metrics in test_results.items()
        }
    }
    
    metadata_path = get_models_dir() / 'training_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Training metadata saved to: {metadata_path}")
    
    # Final Summary
    print("\n\n" + "="*80)
    print("PRODUCTION TRAINING COMPLETE - SUMMARY")
    print("="*80)
    print(f"\nTraining Duration: {training_time/60:.2f} minutes")
    print(f"\nModels Trained: {len(models)}")
    for name in models.keys():
        print(f"  - {name}")
    
    print(f"\nBest Overall Model: {best_model_name}")
    print(f"  Accuracy:   {test_results[best_model_name]['accuracy']:.4f}")
    print(f"  Precision:  {test_results[best_model_name]['precision']:.4f}")
    print(f"  Recall:     {test_results[best_model_name]['recall']:.4f}")
    print(f"  F1-Score:   {test_results[best_model_name]['f1_score']:.4f}")
    print(f"  ROC-AUC:    {test_results[best_model_name]['roc_auc']:.4f}")
    
    print("\nAll models and results saved to: " + str(get_models_dir()))
    print("Preprocessed data saved to: data/processed/")
    
    print("\nYou can now run the web application:")
    print("  python webapp/app.py")
    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

