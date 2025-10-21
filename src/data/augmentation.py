"""Data augmentation techniques for Parkinson's Disease detection."""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.utils import check_random_state

from ..utils.config import Config


class DataAugmentor:
    """Apply various data augmentation techniques to training data."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the data augmentor.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        self.aug_config = self.config.config.get('augmentation', {})
        self.enabled = self.aug_config.get('enabled', True)
        self.random_state = self.config.random_state
        self.rng = check_random_state(self.random_state)
    
    def add_gaussian_noise(
        self,
        X: np.ndarray,
        noise_level: Optional[float] = None
    ) -> np.ndarray:
        """
        Add Gaussian noise to features.
        
        Args:
            X: Input features
            noise_level: Standard deviation of noise as fraction of feature std
            
        Returns:
            Augmented features
        """
        if noise_level is None:
            noise_level = self.aug_config.get('noise_level', 0.05)
        
        # Calculate feature-wise standard deviation
        feature_std = np.std(X, axis=0)
        
        # Generate noise
        noise = self.rng.normal(0, 1, X.shape)
        noise = noise * feature_std * noise_level
        
        # Add noise to features
        X_augmented = X + noise
        
        return X_augmented
    
    def apply_smote(
        self,
        X: np.ndarray,
        y: np.ndarray,
        k_neighbors: Optional[int] = None,
        sampling_strategy: str = 'auto'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply SMOTE (Synthetic Minority Over-sampling Technique).
        
        Args:
            X: Input features
            y: Target labels
            k_neighbors: Number of nearest neighbors
            sampling_strategy: Sampling strategy
            
        Returns:
            Tuple of (augmented features, augmented labels)
        """
        if k_neighbors is None:
            k_neighbors = self.aug_config.get('smote_k_neighbors', 5)
        
        # Ensure k_neighbors is not larger than minority class size
        minority_count = min(np.sum(y == 0), np.sum(y == 1))
        k_neighbors = min(k_neighbors, minority_count - 1)
        
        if k_neighbors < 1:
            print(f"Warning: Not enough samples for SMOTE (k_neighbors={k_neighbors}). Skipping.")
            return X, y
        
        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=k_neighbors,
            random_state=self.random_state
        )
        
        try:
            X_resampled, y_resampled = smote.fit_resample(X, y)
            return X_resampled, y_resampled
        except ValueError as e:
            print(f"Warning: SMOTE failed: {e}. Returning original data.")
            return X, y
    
    def apply_adasyn(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_neighbors: Optional[int] = None,
        sampling_strategy: str = 'auto'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply ADASYN (Adaptive Synthetic Sampling).
        
        Args:
            X: Input features
            y: Target labels
            n_neighbors: Number of nearest neighbors
            sampling_strategy: Sampling strategy
            
        Returns:
            Tuple of (augmented features, augmented labels)
        """
        if n_neighbors is None:
            n_neighbors = self.aug_config.get('adasyn_n_neighbors', 5)
        
        # Ensure n_neighbors is not larger than minority class size
        minority_count = min(np.sum(y == 0), np.sum(y == 1))
        n_neighbors = min(n_neighbors, minority_count - 1)
        
        if n_neighbors < 1:
            print(f"Warning: Not enough samples for ADASYN (n_neighbors={n_neighbors}). Skipping.")
            return X, y
        
        adasyn = ADASYN(
            sampling_strategy=sampling_strategy,
            n_neighbors=n_neighbors,
            random_state=self.random_state
        )
        
        try:
            X_resampled, y_resampled = adasyn.fit_resample(X, y)
            return X_resampled, y_resampled
        except ValueError as e:
            print(f"Warning: ADASYN failed: {e}. Returning original data.")
            return X, y
    
    def apply_mixup(
        self,
        X: np.ndarray,
        y: np.ndarray,
        alpha: Optional[float] = None,
        n_samples: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply Mixup augmentation by interpolating between samples.
        
        Args:
            X: Input features
            y: Target labels
            alpha: Beta distribution parameter for mixing
            n_samples: Number of mixed samples to generate
            
        Returns:
            Tuple of (augmented features, augmented labels)
        """
        if alpha is None:
            alpha = self.aug_config.get('mixup_alpha', 0.2)
        
        if n_samples is None:
            n_samples = len(X) // 2
        
        # Generate random pairs
        n = len(X)
        indices_a = self.rng.choice(n, n_samples, replace=True)
        indices_b = self.rng.choice(n, n_samples, replace=True)
        
        # Generate mixing coefficients
        lam = self.rng.beta(alpha, alpha, n_samples)
        
        # Mix features
        X_mixed = np.array([
            lam[i] * X[indices_a[i]] + (1 - lam[i]) * X[indices_b[i]]
            for i in range(n_samples)
        ])
        
        # For labels, use the majority class from the mix
        y_mixed = np.array([
            y[indices_a[i]] if lam[i] > 0.5 else y[indices_b[i]]
            for i in range(n_samples)
        ])
        
        # Concatenate with original data
        X_augmented = np.vstack([X, X_mixed])
        y_augmented = np.concatenate([y, y_mixed])
        
        return X_augmented, y_augmented
    
    def apply_feature_perturbation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_samples: Optional[int] = None,
        perturbation_ratio: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply feature perturbation with clinical constraints.
        
        Args:
            X: Input features
            y: Target labels
            n_samples: Number of perturbed samples to generate
            perturbation_ratio: Ratio of features to perturb
            
        Returns:
            Tuple of (augmented features, augmented labels)
        """
        if n_samples is None:
            n_samples = len(X) // 2
        
        n_features = X.shape[1]
        n_perturb = max(1, int(n_features * perturbation_ratio))
        
        X_perturbed = []
        y_perturbed = []
        
        for _ in range(n_samples):
            # Randomly select a sample
            idx = self.rng.choice(len(X))
            sample = X[idx].copy()
            
            # Randomly select features to perturb
            feature_indices = self.rng.choice(n_features, n_perturb, replace=False)
            
            # Perturb selected features by small random amounts
            for feat_idx in feature_indices:
                # Calculate feature statistics
                feat_mean = np.mean(X[:, feat_idx])
                feat_std = np.std(X[:, feat_idx])
                
                # Add small perturbation
                perturbation = self.rng.normal(0, feat_std * 0.1)
                sample[feat_idx] += perturbation
            
            X_perturbed.append(sample)
            y_perturbed.append(y[idx])
        
        X_perturbed = np.array(X_perturbed)
        y_perturbed = np.array(y_perturbed)
        
        # Concatenate with original data
        X_augmented = np.vstack([X, X_perturbed])
        y_augmented = np.concatenate([y, y_perturbed])
        
        return X_augmented, y_augmented
    
    def augment_training_data(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        methods: Optional[list] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply comprehensive augmentation pipeline.
        
        Args:
            X_train: Training features
            y_train: Training labels
            methods: List of augmentation methods to apply
                     Options: 'noise', 'smote', 'adasyn', 'mixup', 'perturbation'
            
        Returns:
            Tuple of (augmented features, augmented labels)
        """
        if not self.enabled:
            print("Data augmentation is disabled in config.")
            return X_train, y_train
        
        if methods is None:
            methods = ['noise', 'smote', 'mixup']
        
        print("\n" + "="*60)
        print("Applying Data Augmentation")
        print("="*60)
        print(f"Original training set: {X_train.shape[0]} samples")
        print(f"Augmentation methods: {methods}\n")
        
        X_augmented = X_train.copy()
        y_augmented = y_train.copy()
        
        # Apply each augmentation method
        if 'smote' in methods:
            print("Applying SMOTE...")
            X_augmented, y_augmented = self.apply_smote(X_augmented, y_augmented)
            print(f"  After SMOTE: {X_augmented.shape[0]} samples")
        
        if 'adasyn' in methods:
            print("Applying ADASYN...")
            X_augmented, y_augmented = self.apply_adasyn(X_augmented, y_augmented)
            print(f"  After ADASYN: {X_augmented.shape[0]} samples")
        
        if 'mixup' in methods:
            print("Applying Mixup...")
            n_mixup = int(len(X_train) * 0.3)  # Generate 30% additional samples
            X_augmented, y_augmented = self.apply_mixup(
                X_augmented, y_augmented, n_samples=n_mixup
            )
            print(f"  After Mixup: {X_augmented.shape[0]} samples")
        
        if 'perturbation' in methods:
            print("Applying Feature Perturbation...")
            n_perturb = int(len(X_train) * 0.2)  # Generate 20% additional samples
            X_augmented, y_augmented = self.apply_feature_perturbation(
                X_augmented, y_augmented, n_samples=n_perturb
            )
            print(f"  After Perturbation: {X_augmented.shape[0]} samples")
        
        if 'noise' in methods:
            print("Adding Gaussian Noise...")
            X_augmented = self.add_gaussian_noise(X_augmented)
            print(f"  Noise added to all samples")
        
        # Calculate augmentation statistics
        original_counts = np.bincount(y_train.astype(int))
        augmented_counts = np.bincount(y_augmented.astype(int))
        
        print(f"\nAugmentation Summary:")
        print(f"  Original - Healthy: {original_counts[0]}, PD: {original_counts[1]}")
        print(f"  Augmented - Healthy: {augmented_counts[0]}, PD: {augmented_counts[1]}")
        print(f"  Total samples: {X_train.shape[0]} → {X_augmented.shape[0]}")
        print(f"  Augmentation factor: {X_augmented.shape[0] / X_train.shape[0]:.2f}x")
        print("="*60 + "\n")
        
        return X_augmented, y_augmented


if __name__ == "__main__":
    # Test data augmentation
    print("Testing Data Augmentation Module")
    print("="*60)
    
    # Create synthetic test data
    np.random.seed(42)
    X_train = np.random.randn(100, 10)
    y_train = np.array([0] * 30 + [1] * 70)
    
    # Initialize augmentor
    augmentor = DataAugmentor()
    
    # Test each augmentation method
    print("\n1. Testing Gaussian Noise:")
    X_noise = augmentor.add_gaussian_noise(X_train)
    print(f"   Shape: {X_noise.shape}")
    
    print("\n2. Testing SMOTE:")
    X_smote, y_smote = augmentor.apply_smote(X_train, y_train)
    print(f"   Shape: {X_smote.shape}")
    print(f"   Class distribution: {np.bincount(y_smote.astype(int))}")
    
    print("\n3. Testing Mixup:")
    X_mixup, y_mixup = augmentor.apply_mixup(X_train, y_train, n_samples=20)
    print(f"   Shape: {X_mixup.shape}")
    
    print("\n4. Testing Full Pipeline:")
    X_aug, y_aug = augmentor.augment_training_data(
        X_train, y_train,
        methods=['noise', 'smote', 'mixup']
    )
    
    print("\nData Augmentation Module Test Complete!")

