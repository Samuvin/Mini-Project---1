"""Test suite for model training and evaluation."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.logistic_regression import LogisticRegressionModel
from src.models.svm_model import SVMModel
from src.models.random_forest import RandomForestModel
from src.models.gradient_boosting import XGBoostModel, LightGBMModel
from src.models.ensemble import EnsembleModel
from src.evaluation.metrics import ModelEvaluator
from src.evaluation.stability import StabilityAnalyzer
from src.data.augmentation import DataAugmentor
from src.utils.config import Config


class TestLogisticRegression:
    """Test Logistic Regression model."""
    
    @pytest.fixture
    def model(self):
        """Create a LogisticRegression model instance."""
        return LogisticRegressionModel()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        X_train = np.random.randn(100, 20)
        y_train = np.random.randint(0, 2, 100)
        X_test = np.random.randn(30, 20)
        y_test = np.random.randint(0, 2, 30)
        return X_train, y_train, X_test, y_test
    
    def test_model_initialization(self, model):
        """Test model initialization."""
        assert model is not None
        assert model.config is not None
        assert model.model is None  # Not trained yet
    
    def test_create_model(self, model):
        """Test model creation."""
        lr = model.create_model()
        assert lr is not None
        assert hasattr(lr, 'fit')
        assert hasattr(lr, 'predict')
    
    def test_train_and_predict(self, model, sample_data):
        """Test model training and prediction."""
        X_train, y_train, X_test, y_test = sample_data
        
        # Train model (with reduced parameter grid for speed)
        model.model_config['C'] = [0.1, 1.0]
        model.model_config['penalty'] = ['l2']
        model.train(X_train, y_train)
        
        assert model.model is not None
        
        # Make predictions
        predictions = model.predict(X_test)
        assert len(predictions) == len(y_test)
        assert set(predictions).issubset({0, 1})
        
        # Get probabilities
        proba = model.predict_proba(X_test)
        assert proba.shape == (len(y_test), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)


class TestSVM:
    """Test SVM model."""
    
    @pytest.fixture
    def model(self):
        """Create an SVM model instance."""
        return SVMModel()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        X_train = np.random.randn(100, 20)
        y_train = np.random.randint(0, 2, 100)
        X_test = np.random.randn(30, 20)
        y_test = np.random.randint(0, 2, 30)
        return X_train, y_train, X_test, y_test
    
    def test_model_initialization(self, model):
        """Test model initialization."""
        assert model is not None
        assert model.config is not None
        assert model.model is None
    
    def test_create_model(self, model):
        """Test model creation."""
        svm = model.create_model()
        assert svm is not None
        assert hasattr(svm, 'fit')
        assert hasattr(svm, 'predict')
    
    def test_train_and_predict(self, model, sample_data):
        """Test model training and prediction."""
        X_train, y_train, X_test, y_test = sample_data
        
        # Train model (with reduced parameter grid for speed)
        model.model_config['kernel'] = ['linear', 'rbf']
        model.model_config['C'] = [0.1, 1.0]
        model.model_config['gamma'] = ['scale']
        model.train(X_train, y_train)
        
        assert model.model is not None
        
        # Make predictions
        predictions = model.predict(X_test)
        assert len(predictions) == len(y_test)
        assert set(predictions).issubset({0, 1})
        
        # Get probabilities
        proba = model.predict_proba(X_test)
        assert proba.shape == (len(y_test), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)


class TestModelEvaluator:
    """Test model evaluation functionality."""
    
    @pytest.fixture
    def evaluator(self):
        """Create a ModelEvaluator instance."""
        return ModelEvaluator()
    
    @pytest.fixture
    def predictions(self):
        """Create sample predictions for testing."""
        np.random.seed(42)
        y_true = np.array([0, 1, 1, 0, 1, 1, 0, 0, 1, 0] * 10)
        y_pred = y_true.copy()
        y_pred[::10] = 1 - y_pred[::10]  # Add some errors
        y_pred_proba = np.random.rand(len(y_true))
        y_pred_proba[y_true == 1] += 0.3
        y_pred_proba = np.clip(y_pred_proba, 0, 1)
        return y_true, y_pred, y_pred_proba
    
    def test_evaluator_initialization(self, evaluator):
        """Test evaluator initialization."""
        assert evaluator is not None
        assert evaluator.config is not None
    
    def test_calculate_metrics(self, evaluator, predictions):
        """Test metric calculation."""
        y_true, y_pred, y_pred_proba = predictions
        
        metrics = evaluator.calculate_metrics(y_true, y_pred, y_pred_proba)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'roc_auc' in metrics
        
        # Check metric ranges
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1_score'] <= 1
        assert 0 <= metrics['roc_auc'] <= 1
    
    def test_confusion_matrix_values(self, evaluator, predictions):
        """Test confusion matrix calculations."""
        y_true, y_pred, y_pred_proba = predictions
        
        metrics = evaluator.calculate_metrics(y_true, y_pred, y_pred_proba)
        
        assert 'true_positives' in metrics
        assert 'true_negatives' in metrics
        assert 'false_positives' in metrics
        assert 'false_negatives' in metrics
        
        # Check that counts sum to total
        total = (metrics['true_positives'] + metrics['true_negatives'] + 
                metrics['false_positives'] + metrics['false_negatives'])
        assert total == len(y_true)


class TestRandomForest:
    """Test Random Forest model."""
    
    @pytest.fixture
    def model(self):
        """Create a Random Forest model instance."""
        return RandomForestModel()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        X_train = np.random.randn(100, 20)
        y_train = np.random.randint(0, 2, 100)
        X_test = np.random.randn(30, 20)
        y_test = np.random.randint(0, 2, 30)
        return X_train, y_train, X_test, y_test
    
    def test_model_initialization(self, model):
        """Test model initialization."""
        assert model is not None
        assert model.config is not None
        assert model.model is None
    
    def test_create_model(self, model):
        """Test model creation."""
        rf = model.create_model()
        assert rf is not None
        assert hasattr(rf, 'fit')
        assert hasattr(rf, 'predict')
    
    def test_train_and_predict(self, model, sample_data):
        """Test model training and prediction."""
        X_train, y_train, X_test, y_test = sample_data
        
        # Train model (with reduced parameter grid for speed)
        model.model_config['n_estimators'] = [50, 100]
        model.model_config['max_depth'] = [5, 10]
        model.model_config['min_samples_split'] = [2]
        model.train(X_train, y_train, search_method='grid')
        
        assert model.model is not None
        
        # Make predictions
        predictions = model.predict(X_test)
        assert len(predictions) == len(y_test)
        assert set(predictions).issubset({0, 1})


class TestXGBoost:
    """Test XGBoost model."""
    
    @pytest.fixture
    def model(self):
        """Create an XGBoost model instance."""
        return XGBoostModel()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        X_train = np.random.randn(100, 20)
        y_train = np.random.randint(0, 2, 100)
        X_test = np.random.randn(30, 20)
        y_test = np.random.randint(0, 2, 30)
        return X_train, y_train, X_test, y_test
    
    def test_model_initialization(self, model):
        """Test model initialization."""
        assert model is not None
        assert model.model is None
    
    def test_create_model(self, model):
        """Test model creation."""
        xgb = model.create_model()
        assert xgb is not None
        assert hasattr(xgb, 'fit')
    
    def test_train_and_predict(self, model, sample_data):
        """Test model training and prediction."""
        X_train, y_train, X_test, y_test = sample_data
        
        # Train model with minimal grid
        model.model_config['n_estimators'] = [50]
        model.model_config['max_depth'] = [3]
        model.model_config['learning_rate'] = [0.1]
        model.train(X_train, y_train, search_method='grid')
        
        assert model.model is not None
        
        predictions = model.predict(X_test)
        assert len(predictions) == len(y_test)


class TestDataAugmentation:
    """Test data augmentation module."""
    
    @pytest.fixture
    def augmentor(self):
        """Create a DataAugmentor instance."""
        return DataAugmentor()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = np.array([0] * 30 + [1] * 70)
        return X, y
    
    def test_augmentor_initialization(self, augmentor):
        """Test augmentor initialization."""
        assert augmentor is not None
        assert augmentor.config is not None
    
    def test_gaussian_noise(self, augmentor, sample_data):
        """Test Gaussian noise augmentation."""
        X, y = sample_data
        X_noisy = augmentor.add_gaussian_noise(X)
        
        assert X_noisy.shape == X.shape
        assert not np.array_equal(X_noisy, X)  # Should be different
    
    def test_smote(self, augmentor, sample_data):
        """Test SMOTE augmentation."""
        X, y = sample_data
        X_smote, y_smote = augmentor.apply_smote(X, y)
        
        # Should have more samples
        assert len(X_smote) >= len(X)
        assert len(y_smote) >= len(y)
    
    def test_mixup(self, augmentor, sample_data):
        """Test mixup augmentation."""
        X, y = sample_data
        X_mixup, y_mixup = augmentor.apply_mixup(X, y, n_samples=20)
        
        # Should have additional samples
        assert len(X_mixup) == len(X) + 20
        assert len(y_mixup) == len(y) + 20
    
    def test_augmentation_no_data_leakage(self, augmentor, sample_data):
        """Test that augmentation doesn't leak data."""
        X, y = sample_data
        X_aug, y_aug = augmentor.augment_training_data(X, y)
        
        # Original data should be included in augmented data
        assert len(X_aug) >= len(X)
        assert len(y_aug) >= len(y)


class TestStabilityAnalyzer:
    """Test stability analysis module."""
    
    @pytest.fixture
    def analyzer(self):
        """Create a StabilityAnalyzer instance."""
        return StabilityAnalyzer()
    
    @pytest.fixture
    def cv_scores(self):
        """Create sample CV scores."""
        np.random.seed(42)
        scores_a = np.random.normal(0.85, 0.05, 5)
        scores_b = np.random.normal(0.80, 0.08, 5)
        return scores_a, scores_b
    
    def test_analyzer_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer is not None
        assert analyzer.config is not None
    
    def test_cv_statistics(self, analyzer, cv_scores):
        """Test CV statistics calculation."""
        scores_a, _ = cv_scores
        stats = analyzer.calculate_cv_statistics(scores_a)
        
        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert 'cv_coefficient' in stats
        
        # Check reasonable values
        assert 0 <= stats['mean'] <= 1
        assert stats['std'] >= 0
    
    def test_confidence_interval(self, analyzer, cv_scores):
        """Test confidence interval calculation."""
        scores_a, _ = cv_scores
        ci_lower, ci_upper = analyzer.calculate_confidence_interval(scores_a)
        
        assert ci_lower < ci_upper
        assert ci_lower <= np.mean(scores_a) <= ci_upper
    
    def test_paired_ttest(self, analyzer, cv_scores):
        """Test paired t-test."""
        scores_a, scores_b = cv_scores
        results = analyzer.perform_paired_ttest(scores_a, scores_b)
        
        assert 'p_value' in results
        assert 't_statistic' in results
        assert 'cohens_d' in results
        assert 'is_significant' in results
        assert 'winner' in results
        
        # Check reasonable p-value
        assert 0 <= results['p_value'] <= 1


class TestEnsembleModel:
    """Test ensemble model."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        X_train = np.random.randn(100, 20)
        y_train = np.random.randint(0, 2, 100)
        X_test = np.random.randn(30, 20)
        y_test = np.random.randint(0, 2, 30)
        return X_train, y_train, X_test, y_test
    
    def test_voting_ensemble_creation(self, sample_data):
        """Test voting ensemble creation."""
        X_train, y_train, X_test, y_test = sample_data
        
        # Create simple base models
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        
        estimators = [
            ('lr', LogisticRegression(random_state=42)),
            ('dt', DecisionTreeClassifier(random_state=42))
        ]
        
        ensemble = EnsembleModel()
        voting_clf = ensemble.create_voting_ensemble(estimators)
        
        assert voting_clf is not None
        assert hasattr(voting_clf, 'fit')
        assert hasattr(voting_clf, 'predict')
    
    def test_stacking_ensemble_creation(self, sample_data):
        """Test stacking ensemble creation."""
        X_train, y_train, X_test, y_test = sample_data
        
        # Create simple base models
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        
        estimators = [
            ('lr', LogisticRegression(random_state=42)),
            ('dt', DecisionTreeClassifier(random_state=42))
        ]
        
        ensemble = EnsembleModel()
        stacking_clf = ensemble.create_stacking_ensemble(estimators)
        
        assert stacking_clf is not None
        assert hasattr(stacking_clf, 'fit')
        assert hasattr(stacking_clf, 'predict')


class TestNestedCV:
    """Test nested cross-validation."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        X = np.random.randn(100, 20)
        y = np.random.randint(0, 2, 100)
        return X, y
    
    def test_nested_cv(self, sample_data):
        """Test nested cross-validation."""
        X, y = sample_data
        
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GridSearchCV
        
        # Create estimator with grid search
        param_grid = {'C': [0.1, 1.0]}
        estimator = GridSearchCV(
            LogisticRegression(random_state=42),
            param_grid,
            cv=3
        )
        
        evaluator = ModelEvaluator()
        scores, stats = evaluator.nested_cross_validation(
            estimator, X, y,
            inner_cv=3,
            outer_cv=3
        )
        
        assert len(scores) == 3  # outer_cv folds
        assert 'mean' in stats
        assert 'std' in stats
        assert 0 <= stats['mean'] <= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

