"""Complete training pipeline for Parkinson's Disease Detection System."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data.data_loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.models.logistic_regression import LogisticRegressionModel
from src.models.svm_model import SVMModel
from src.evaluation.metrics import ModelEvaluator
from src.utils.config import Config, get_models_dir

import pandas as pd


def main():
    """Main training pipeline."""
    print("\n" + "="*80)
    print("PARKINSON'S DISEASE DETECTION SYSTEM - TRAINING PIPELINE")
    print("="*80 + "\n")
    
    # Initialize configuration
    config = Config()
    print("Configuration loaded successfully!\n")
    
    # Step 1: Load Data
    print("Step 1: Loading multimodal datasets...")
    print("-" * 80)
    loader = DataLoader(config)
    
    # Load ONLY speech data for training the speech model (22 features)
    # Handwriting and gait models are trained separately with their own scripts
    X, y = loader.load_speech_data()
    print(f"\n✓ Speech data loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Training speech model for multi-model ensemble\n")
    
    # Step 2: Preprocess Data
    print("Step 2: Preprocessing data...")
    print("-" * 80)
    preprocessor = DataPreprocessor(config)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.preprocess_pipeline(
        X, y, 
        remove_outliers=False,
        balance_classes=True,
        save=True
    )
    
    # Step 3: Train Logistic Regression Baseline
    print("\n\nStep 3: Training Logistic Regression Baseline...")
    print("-" * 80)
    lr_model = LogisticRegressionModel(config)
    lr_history = lr_model.train(X_train, y_train, X_val, y_val)
    
    # Evaluate LR on test set
    lr_test_metrics = lr_model.evaluate(X_test, y_test)
    
    # Save LR model
    lr_model.save_model('logistic_regression_model.joblib')
    
    # Step 4: Train SVM with Kernel Optimization
    print("\n\nStep 4: Training SVM with Kernel Optimization...")
    print("-" * 80)
    svm_model = SVMModel(config)
    svm_history = svm_model.train(X_train, y_train, X_val, y_val, search_method='grid')
    
    # Print kernel comparison
    svm_model.print_kernel_comparison()
    
    # Evaluate SVM on test set
    svm_test_metrics = svm_model.evaluate(X_test, y_test)
    
    # Save SVM model
    svm_model.save_model('svm_model.joblib')
    
    # Step 5: Compare Models
    print("\n\nStep 5: Model Comparison...")
    print("-" * 80)
    
    evaluator = ModelEvaluator(config)
    comparison = evaluator.compare_models(
        {
            'Logistic Regression': lr_test_metrics,
            'SVM (Optimized)': svm_test_metrics
        },
        save_path=get_models_dir() / 'model_comparison.png'
    )
    
    print("\nModel Comparison Results:")
    print(comparison.to_string())
    
    # Save best model
    if svm_test_metrics['accuracy'] > lr_test_metrics['accuracy']:
        print("\n\nSVM achieved better accuracy. Saving as best_model.joblib")
        svm_model.save_model('best_model.joblib')
        best_model_name = 'SVM'
        best_metrics = svm_test_metrics
    else:
        print("\n\nLogistic Regression achieved better accuracy. Saving as best_model.joblib")
        lr_model.save_model('best_model.joblib')
        best_model_name = 'Logistic Regression'
        best_metrics = lr_test_metrics
    
    # Step 6: Final Summary
    print("\n\n" + "="*80)
    print("TRAINING COMPLETE - SUMMARY")
    print("="*80)
    print(f"\nBest Model: {best_model_name}")
    print(f"Accuracy:   {best_metrics['accuracy']:.4f}")
    print(f"Precision:  {best_metrics['precision']:.4f}")
    print(f"Recall:     {best_metrics['recall']:.4f}")
    print(f"F1-Score:   {best_metrics['f1_score']:.4f}")
    print(f"ROC-AUC:    {best_metrics['roc_auc']:.4f}")
    print("\nModels saved to: " + str(get_models_dir()))
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

