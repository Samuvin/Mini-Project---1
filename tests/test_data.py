"""Test suite for data loading and preprocessing."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.utils.config import Config


class TestDataLoader:
    """Test data loading functionality."""
    
    @pytest.fixture
    def loader(self):
        """Create a DataLoader instance."""
        return DataLoader()
    
    def test_loader_initialization(self, loader):
        """Test loader initialization."""
        assert loader is not None
        assert loader.config is not None
    
    def test_load_speech_data(self, loader):
        """Test loading speech data."""
        X, y = loader.load_speech_data()
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y)
        assert len(X) > 0
        assert set(y.unique()).issubset({0, 1})
    
    def test_load_handwriting_data(self, loader):
        """Test loading handwriting data."""
        X, y = loader.load_handwriting_data()
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y)
        assert set(y.unique()).issubset({0, 1})
    
    def test_load_gait_data(self, loader):
        """Test loading gait data."""
        X, y = loader.load_gait_data()
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y)
        assert set(y.unique()).issubset({0, 1})
    
    def test_load_all_modalities(self, loader):
        """Test loading combined multimodal data."""
        X, y = loader.load_all_modalities()
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y)
        assert X.shape[1] > 20  # Should have features from all modalities


class TestDataPreprocessor:
    """Test data preprocessing functionality."""
    
    @pytest.fixture
    def preprocessor(self):
        """Create a DataPreprocessor instance."""
        return DataPreprocessor()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 10))
        y = pd.Series(np.random.randint(0, 2, 100))
        return X, y
    
    def test_preprocessor_initialization(self, preprocessor):
        """Test preprocessor initialization."""
        assert preprocessor is not None
        assert preprocessor.config is not None
        assert preprocessor.scaler is not None
    
    def test_handle_missing_values(self, preprocessor, sample_data):
        """Test missing value handling."""
        X, _ = sample_data
        X_with_nan = X.copy()
        X_with_nan.iloc[0, 0] = np.nan
        
        X_filled = preprocessor.handle_missing_values(X_with_nan)
        assert X_filled.isnull().sum().sum() == 0
    
    def test_split_data(self, preprocessor, sample_data):
        """Test data splitting."""
        X, y = sample_data
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)
        
        assert len(X_train) + len(X_val) + len(X_test) == len(X)
        assert len(y_train) + len(y_val) + len(y_test) == len(y)
        assert len(X_train) > len(X_val)
        assert len(X_train) > len(X_test)
    
    def test_normalize_features(self, preprocessor, sample_data):
        """Test feature normalization."""
        X, _ = sample_data
        X_train, X_val, X_test = X.iloc[:70], X.iloc[70:85], X.iloc[85:]
        
        X_train_norm, X_val_norm, X_test_norm = preprocessor.normalize_features(
            X_train, X_val, X_test
        )
        
        # Check shapes
        assert X_train_norm.shape == X_train.shape
        assert X_val_norm.shape == X_val.shape
        assert X_test_norm.shape == X_test.shape
        
        # Check normalization (mean should be close to 0, std close to 1)
        assert np.abs(X_train_norm.mean()) < 0.1
        assert np.abs(X_train_norm.std() - 1.0) < 0.1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

