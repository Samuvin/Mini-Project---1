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
from src.evaluation.metrics import ModelEvaluator
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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

