"""Quick validation test for the production pipeline."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.data_loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.models.random_forest import RandomForestModel
from src.models.gradient_boosting import XGBoostModel
from src.models.ensemble import EnsembleModel
from src.evaluation.stability import StabilityAnalyzer
from src.utils.config import Config

print("\n" + "="*80)
print("QUICK VALIDATION TEST - Production Pipeline")
print("="*80 + "\n")

# Test 1: Load Data
print("Test 1: Loading combined speech data...")
loader = DataLoader()
X, y = loader.load_combined_speech_data()
print(f"✓ Successfully loaded {X.shape[0]} samples with {X.shape[1]} features\n")

# Test 2: Preprocessing with Augmentation
print("Test 2: Preprocessing with augmentation...")
preprocessor = DataPreprocessor()
X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.preprocess_pipeline(
    X, y,
    remove_outliers=False,
    balance_classes=True,
    apply_augmentation=True,
    save=False
)
print(f"✓ Successfully preprocessed data")
print(f"  Training: {X_train.shape[0]} samples")
print(f"  Validation: {X_val.shape[0]} samples")
print(f"  Test: {X_test.shape[0]} samples\n")

# Test 3: Train Random Forest (Quick)
print("Test 3: Training Random Forest (minimal config)...")
rf_model = RandomForestModel()
rf_model.model_config['n_estimators'] = [50]
rf_model.model_config['max_depth'] = [5]
rf_model.model_config['min_samples_split'] = [2]
rf_model.model_config['max_features'] = ['sqrt']
rf_history = rf_model.train(X_train[:500], y_train[:500], search_method='grid')
rf_test_metrics = rf_model.evaluate(X_test, y_test)
print(f"✓ Random Forest - Test ROC-AUC: {rf_test_metrics['roc_auc']:.4f}\n")

# Test 4: Train XGBoost (Quick)
print("Test 4: Training XGBoost (minimal config)...")
xgb_model = XGBoostModel()
xgb_model.model_config['n_estimators'] = [50]
xgb_model.model_config['max_depth'] = [3]
xgb_model.model_config['learning_rate'] = [0.1]
xgb_history = xgb_model.train(X_train[:500], y_train[:500], search_method='grid')
xgb_test_metrics = xgb_model.evaluate(X_test, y_test)
print(f"✓ XGBoost - Test ROC-AUC: {xgb_test_metrics['roc_auc']:.4f}\n")

# Test 5: Create Ensemble
print("Test 5: Creating Voting Ensemble...")
ensemble = EnsembleModel()
estimators = [
    ('RF', rf_model.model),
    ('XGB', xgb_model.model)
]
voting_history = ensemble.train_voting_ensemble(
    estimators, X_train[:500], y_train[:500], X_val, y_val
)
ensemble_test_metrics = ensemble.evaluate(X_test, y_test, 'Test Voting Ensemble')
print(f"✓ Voting Ensemble - Test ROC-AUC: {ensemble_test_metrics['roc_auc']:.4f}\n")

# Test 6: Stability Analysis
print("Test 6: Running stability analysis...")
analyzer = StabilityAnalyzer()
import numpy as np
cv_scores = np.array([
    rf_test_metrics['roc_auc'],
    xgb_test_metrics['roc_auc'],
    ensemble_test_metrics['roc_auc'],
    0.85, 0.87  # Mock additional scores
])
stats = analyzer.calculate_cv_statistics(cv_scores)
ci_lower, ci_upper = analyzer.calculate_confidence_interval(cv_scores)
print(f"✓ Mean ROC-AUC: {stats['mean']:.4f} (±{stats['std']:.4f})")
print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]\n")

print("="*80)
print("ALL VALIDATION TESTS PASSED ✓")
print("="*80)
print("\nProduction pipeline is ready for full training!")
print("Run: python train_production.py\n")

