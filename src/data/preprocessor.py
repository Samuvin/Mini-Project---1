"""Data preprocessing utilities for Parkinson's Disease prediction."""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from ..utils.config import Config, get_processed_data_dir, ensure_dir_exists


class DataPreprocessor:
    """Preprocess data for machine learning models."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the data preprocessor.
        
        Args:
            config: Configuration object. If None, creates a new Config instance.
        """
        self.config = config or Config()
        self.scaler = StandardScaler()
        self.smote = None
        self.augmentor = None  # Lazy initialization if needed
        self.processed_dir = get_processed_data_dir()
        ensure_dir_exists(self.processed_dir)
    
    def handle_missing_values(
        self, 
        X: pd.DataFrame, 
        strategy: str = 'mean'
    ) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            X: Input features DataFrame
            strategy: Strategy for handling missing values ('mean', 'median', 'forward_fill')
            
        Returns:
            DataFrame with missing values handled
        """
        if X.isnull().sum().sum() == 0:
            print("No missing values found.")
            return X
        
        print(f"Handling missing values using strategy: {strategy}")
        
        if strategy == 'mean':
            X = X.fillna(X.mean())
        elif strategy == 'median':
            X = X.fillna(X.median())
        elif strategy == 'forward_fill':
            X = X.fillna(method='ffill').fillna(method='bfill')
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        return X
    
    def remove_outliers(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        method: str = 'iqr',
        threshold: float = 3.0
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Remove outliers from the dataset.
        
        Args:
            X: Input features DataFrame
            y: Target labels Series
            method: Method for outlier detection ('iqr' or 'zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            Tuple of (cleaned features, cleaned labels)
        """
        print(f"Removing outliers using {method} method...")
        original_size = len(X)
        
        if method == 'iqr':
            Q1 = X.quantile(0.25)
            Q3 = X.quantile(0.75)
            IQR = Q3 - Q1
            outlier_mask = ~((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))).any(axis=1)
        elif method == 'zscore':
            z_scores = np.abs((X - X.mean()) / X.std())
            outlier_mask = (z_scores < threshold).all(axis=1)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        X_cleaned = X[outlier_mask]
        y_cleaned = y[outlier_mask]
        
        removed = original_size - len(X_cleaned)
        print(f"Removed {removed} outliers ({removed/original_size*100:.2f}%)")
        
        return X_cleaned, y_cleaned
    
    def normalize_features(
        self, 
        X_train: pd.DataFrame,
        X_val: Optional[pd.DataFrame] = None,
        X_test: Optional[pd.DataFrame] = None
    ) -> Tuple[np.ndarray, ...]:
        """
        Normalize features using StandardScaler.
        
        Args:
            X_train: Training features
            X_val: Validation features (optional)
            X_test: Test features (optional)
            
        Returns:
            Tuple of normalized arrays (train, val, test) - only returns provided sets
        """
        print("Normalizing features (StandardScaler: mean=0, std=1)...")
        
        # Fit scaler on training data
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        results = [X_train_scaled]
        
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            results.append(X_val_scaled)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            results.append(X_test_scaled)
        
        return tuple(results)
    
    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        train_size: Optional[float] = None,
        val_size: Optional[float] = None,
        test_size: Optional[float] = None,
        random_state: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            X: Input features
            y: Target labels
            train_size: Proportion for training set (default from config)
            val_size: Proportion for validation set (default from config)
            test_size: Proportion for test set (default from config)
            random_state: Random state for reproducibility (default from config)
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        train_size = train_size or self.config.train_size
        val_size = val_size or self.config.val_size
        test_size = test_size or self.config.test_size
        random_state = random_state or self.config.random_state
        
        # Validate proportions
        total = train_size + val_size + test_size
        if not np.isclose(total, 1.0):
            raise ValueError(f"Split proportions must sum to 1.0, got {total}")
        
        print(f"Splitting data: train={train_size}, val={val_size}, test={test_size}")
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
        
        # Second split: separate train and validation
        val_proportion = val_size / (train_size + val_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_proportion,
            random_state=random_state,
            stratify=y_temp
        )
        
        print(f"Split sizes - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        print(f"Class distribution:")
        print(f"  Train - PD: {sum(y_train==1)}, Healthy: {sum(y_train==0)}")
        print(f"  Val   - PD: {sum(y_val==1)}, Healthy: {sum(y_val==0)}")
        print(f"  Test  - PD: {sum(y_test==1)}, Healthy: {sum(y_test==0)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def balance_classes(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        method: str = 'smote',
        sampling_strategy: str = 'auto'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Balance classes using oversampling techniques.
        
        Args:
            X_train: Training features
            y_train: Training labels
            method: Balancing method ('smote' or 'none')
            sampling_strategy: Sampling strategy for SMOTE
            
        Returns:
            Tuple of (balanced features, balanced labels)
        """
        if method == 'none':
            print("Skipping class balancing.")
            return X_train, y_train
        
        print(f"Balancing classes using {method.upper()}...")
        original_counts = np.bincount(y_train.astype(int))
        print(f"Original distribution - PD: {original_counts[1]}, Healthy: {original_counts[0]}")
        
        if method == 'smote':
            self.smote = SMOTE(
                sampling_strategy=sampling_strategy,
                random_state=self.config.random_state
            )
            X_balanced, y_balanced = self.smote.fit_resample(X_train, y_train)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        new_counts = np.bincount(y_balanced.astype(int))
        print(f"Balanced distribution - PD: {new_counts[1]}, Healthy: {new_counts[0]}")
        
        return X_balanced, y_balanced
    
    def preprocess_pipeline(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        remove_outliers: bool = False,
        balance_classes: bool = True,
        apply_augmentation: bool = True,
        save: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Complete preprocessing pipeline with augmentation support.
        
        Args:
            X: Input features DataFrame
            y: Target labels Series
            remove_outliers: Whether to remove outliers
            balance_classes: Whether to balance classes using SMOTE
            apply_augmentation: Whether to apply data augmentation
            save: Whether to save preprocessed data
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        print("\n" + "="*60)
        print("Starting Preprocessing Pipeline")
        print("="*60 + "\n")
        
        # Step 1: Handle missing values
        X = self.handle_missing_values(X)
        
        # Step 2: Remove outliers (optional)
        if remove_outliers:
            X, y = self.remove_outliers(X, y)
        
        # Step 3: Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        
        # Step 4: Normalize features
        X_train_norm, X_val_norm, X_test_norm = self.normalize_features(
            X_train, X_val, X_test
        )
        
        # Step 5: Balance classes (only on training set)
        if balance_classes and self.config.get('training.use_smote', True):
            X_train_norm, y_train = self.balance_classes(
                X_train_norm,
                y_train.values,
                method='smote'
            )
        else:
            y_train = y_train.values
        
        # Step 6: Apply data augmentation (only on training set)
        if apply_augmentation and self.config.get('augmentation.enabled', False):
            try:
                from .augmentation import DataAugmentor
                if self.augmentor is None:
                    self.augmentor = DataAugmentor(self.config)
                aug_methods = ['noise', 'smote', 'mixup']
                X_train_norm, y_train = self.augmentor.augment_training_data(
                    X_train_norm, y_train, methods=aug_methods
                )
            except ImportError:
                print("Warning: Data augmentation module not found. Skipping augmentation.")
                print("To enable augmentation, create src/data/augmentation.py with DataAugmentor class.")
        
        # Convert to numpy arrays
        y_val = y_val.values
        y_test = y_test.values
        
        # Step 7: Save preprocessed data and scaler
        if save:
            self.save_preprocessed_data(
                X_train_norm, X_val_norm, X_test_norm,
                y_train, y_val, y_test
            )
            self.save_scaler()
        
        print("\n" + "="*60)
        print("Preprocessing Pipeline Complete")
        print("="*60 + "\n")
        
        return X_train_norm, X_val_norm, X_test_norm, y_train, y_val, y_test
    
    def save_preprocessed_data(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray
    ) -> None:
        """
        Save preprocessed data to disk.
        
        Args:
            X_train: Training features
            X_val: Validation features
            X_test: Test features
            y_train: Training labels
            y_val: Validation labels
            y_test: Test labels
        """
        print(f"Saving preprocessed data to {self.processed_dir}...")
        
        np.save(self.processed_dir / 'X_train.npy', X_train)
        np.save(self.processed_dir / 'X_val.npy', X_val)
        np.save(self.processed_dir / 'X_test.npy', X_test)
        np.save(self.processed_dir / 'y_train.npy', y_train)
        np.save(self.processed_dir / 'y_val.npy', y_val)
        np.save(self.processed_dir / 'y_test.npy', y_test)
        
        print("Preprocessed data saved successfully!")
    
    def save_scaler(self) -> None:
        """Save the fitted scaler to the models directory."""
        import joblib
        from ..utils.config import get_models_dir
        
        models_dir = get_models_dir()
        scaler_path = models_dir / 'scaler.joblib'
        
        joblib.dump(self.scaler, scaler_path)
        print(f"Scaler saved to {scaler_path}")
    
    def load_preprocessed_data(
        self
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load preprocessed data from disk.
        
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
            
        Raises:
            FileNotFoundError: If preprocessed data files don't exist
        """
        print(f"Loading preprocessed data from {self.processed_dir}...")
        
        X_train = np.load(self.processed_dir / 'X_train.npy')
        X_val = np.load(self.processed_dir / 'X_val.npy')
        X_test = np.load(self.processed_dir / 'X_test.npy')
        y_train = np.load(self.processed_dir / 'y_train.npy')
        y_val = np.load(self.processed_dir / 'y_val.npy')
        y_test = np.load(self.processed_dir / 'y_test.npy')
        
        print(f"Loaded - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    # Test preprocessing pipeline
    from .data_loader import DataLoader
    
    print("Testing Data Preprocessing Pipeline\n")
    
    # Load data
    loader = DataLoader()
    X, y = loader.load_all_modalities()
    
    # Preprocess
    preprocessor = DataPreprocessor()
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.preprocess_pipeline(
        X, y, remove_outliers=False, balance_classes=True
    )
    
    print("\nFinal shapes:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

