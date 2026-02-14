"""Support Vector Machine model with kernel optimization for Parkinson's Disease prediction."""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import joblib
from pathlib import Path
import time

from ..utils.config import Config, get_models_dir, ensure_dir_exists
from ..evaluation.metrics import ModelEvaluator


class SVMModel:
    """SVM model with kernel optimization and hyperparameter tuning."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the SVM model.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        self.model_config = self.config.get_model_config('svm')
        self.training_config = self.config.get_training_config()
        
        self.model = None
        self.best_model = None
        self.grid_search = None
        self.kernel_models = {}  # Store models for each kernel type
        self.training_history = {}
        
        self.models_dir = get_models_dir()
        ensure_dir_exists(self.models_dir)
    
    def create_model(
        self,
        kernel: str = 'rbf',
        C: float = 1.0,
        gamma: str = 'scale',
        degree: int = 3,
        class_weight: str = 'balanced',
        probability: bool = True
    ) -> SVC:
        """
        Create an SVM model with specified parameters.
        
        Args:
            kernel: Kernel type ('linear', 'rbf', 'poly', 'sigmoid')
            C: Regularization parameter
            gamma: Kernel coefficient
            degree: Degree for polynomial kernel
            class_weight: Class weight strategy
            probability: Whether to enable probability estimates
            
        Returns:
            SVC model instance
        """
        random_state = self.config.random_state
        
        model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            degree=degree,
            class_weight=class_weight,
            probability=probability,
            random_state=random_state
        )
        
        return model
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        search_method: str = 'grid'
    ) -> Dict[str, any]:
        """
        Train the SVM model with kernel optimization and hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            search_method: Search method ('grid' or 'random')
            
        Returns:
            Dictionary containing training results and metrics
        """
        print("\n" + "="*60)
        print("Training SVM Model with Kernel Optimization")
        print("="*60 + "\n")
        
        # Prepare parameter grid for hyperparameter search
        param_grid = {
            'kernel': self.model_config.get('kernel', ['linear', 'rbf', 'poly', 'sigmoid']),
            'C': self.model_config.get('C', [0.1, 1, 10, 100]),
            'gamma': self.model_config.get('gamma', ['scale', 'auto', 0.001, 0.01, 0.1, 1])
        }
        
        # Add degree parameter for polynomial kernel
        if 'poly' in param_grid['kernel']:
            param_grid['degree'] = self.model_config.get('degree', [2, 3, 4])
        
        # Create base model
        base_model = self.create_model()
        
        # Setup hyperparameter search
        cv_folds = self.training_config.get('cv_folds', 5)
        scoring = self.training_config.get('scoring', 'roc_auc')
        n_jobs = self.training_config.get('n_jobs', -1)
        verbose = self.training_config.get('verbose', 1)
        
        print(f"Hyperparameter search method: {search_method.upper()}")
        print(f"Cross-validation folds: {cv_folds}")
        print(f"Parameter grid: {param_grid}")
        print(f"Scoring metric: {scoring}\n")
        
        start_time = time.time()
        
        if search_method == 'grid':
            self.grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                cv=cv_folds,
                scoring=scoring,
                n_jobs=n_jobs,
                verbose=verbose,
                return_train_score=True
            )
        elif search_method == 'random':
            self.grid_search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=param_grid,
                n_iter=50,
                cv=cv_folds,
                scoring=scoring,
                n_jobs=n_jobs,
                verbose=verbose,
                random_state=self.config.random_state,
                return_train_score=True
            )
        else:
            raise ValueError(f"Unknown search method: {search_method}")
        
        # Fit hyperparameter search
        print("Starting hyperparameter optimization...")
        self.grid_search.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        
        # Get best model
        self.best_model = self.grid_search.best_estimator_
        self.model = self.best_model
        
        print("\n" + "-"*60)
        print("Best Parameters:")
        for param, value in self.grid_search.best_params_.items():
            print(f"  {param}: {value}")
        print(f"\nBest Cross-Validation Score ({scoring}): {self.grid_search.best_score_:.4f}")
        print(f"Training time: {training_time:.2f} seconds")
        print("-"*60 + "\n")
        
        # Train individual models for each kernel type (for comparison)
        self._train_kernel_models(X_train, y_train, X_val, y_val)
        
        # Evaluate on training set
        train_pred = self.best_model.predict(X_train)
        train_pred_proba = self.best_model.predict_proba(X_train)[:, 1]
        
        evaluator = ModelEvaluator(self.config)
        train_metrics = evaluator.calculate_metrics(y_train, train_pred, train_pred_proba)
        
        print("Training Set Performance (Best Model):")
        evaluator.print_metrics(train_metrics)
        
        # Evaluate on validation set if provided
        val_metrics = None
        if X_val is not None and y_val is not None:
            val_pred = self.best_model.predict(X_val)
            val_pred_proba = self.best_model.predict_proba(X_val)[:, 1]
            
            val_metrics = evaluator.calculate_metrics(y_val, val_pred, val_pred_proba)
            
            print("\nValidation Set Performance (Best Model):")
            evaluator.print_metrics(val_metrics)
        
        # Store training history
        self.training_history = {
            'best_params': self.grid_search.best_params_,
            'best_score': self.grid_search.best_score_,
            'cv_results': self.grid_search.cv_results_,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'training_time': training_time,
            'kernel_comparison': self._get_kernel_comparison()
        }
        
        print("\n" + "="*60)
        print("SVM Training Complete")
        print("="*60 + "\n")
        
        return self.training_history
    
    def _train_kernel_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> None:
        """
        Train separate models for each kernel type for comparison.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
        """
        print("\n" + "-"*60)
        print("Training Individual Kernel Models for Comparison")
        print("-"*60 + "\n")
        
        kernels = ['linear', 'rbf', 'poly', 'sigmoid']
        evaluator = ModelEvaluator(self.config)
        
        for kernel in kernels:
            print(f"Training {kernel.upper()} kernel...")
            
            # Create and train model with default parameters
            model = self.create_model(kernel=kernel)
            
            try:
                model.fit(X_train, y_train)
                
                # Evaluate on validation set
                if X_val is not None and y_val is not None:
                    val_pred = model.predict(X_val)
                    val_pred_proba = model.predict_proba(X_val)[:, 1]
                    
                    metrics = evaluator.calculate_metrics(y_val, val_pred, val_pred_proba)
                    
                    self.kernel_models[kernel] = {
                        'model': model,
                        'metrics': metrics
                    }
                    
                    print(f"  {kernel.upper()} - Accuracy: {metrics['accuracy']:.4f}, "
                          f"ROC-AUC: {metrics['roc_auc']:.4f}")
                else:
                    self.kernel_models[kernel] = {'model': model, 'metrics': None}
            
            except Exception as e:
                print(f"  Error training {kernel} kernel: {e}")
                self.kernel_models[kernel] = {'model': None, 'metrics': None, 'error': str(e)}
        
        print()
    
    def _get_kernel_comparison(self) -> pd.DataFrame:
        """
        Get comparison of different kernel performances.
        
        Returns:
            DataFrame with kernel comparison
        """
        comparison_data = []
        
        for kernel, data in self.kernel_models.items():
            if data.get('metrics'):
                metrics = data['metrics']
                comparison_data.append({
                    'kernel': kernel,
                    'accuracy': metrics['accuracy'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1_score': metrics['f1_score'],
                    'roc_auc': metrics['roc_auc']
                })
        
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            df = df.sort_values('roc_auc', ascending=False)
            return df
        
        return pd.DataFrame()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input features
            
        Returns:
            Predicted labels
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Input features
            
        Returns:
            Predicted probabilities for each class
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict_proba(X)
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        print("\n" + "="*60)
        print("Evaluating SVM on Test Set")
        print("="*60 + "\n")
        
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)[:, 1]
        
        evaluator = ModelEvaluator(self.config)
        test_metrics = evaluator.calculate_metrics(y_test, y_pred, y_pred_proba)
        
        evaluator.print_metrics(test_metrics)
        
        # Generate visualizations
        evaluator.plot_confusion_matrix(
            y_test, y_pred,
            save_path=self.models_dir / 'svm_confusion_matrix.png'
        )
        
        evaluator.plot_roc_curve(
            y_test, y_pred_proba,
            model_name='SVM (Optimized)',
            save_path=self.models_dir / 'svm_roc_curve.png'
        )
        
        return test_metrics
    
    def save_model(self, filename: Optional[str] = None) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filename: Name of the file to save (default: 'svm_model.joblib')
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        filename = filename or 'svm_model.joblib'
        filepath = self.models_dir / filename
        
        model_data = {
            'model': self.model,
            'best_params': self.grid_search.best_params_ if self.grid_search else None,
            'training_history': self.training_history,
            'config': self.config.config
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to: {filepath}")
    
    def load_model(self, filename: Optional[str] = None) -> None:
        """
        Load a trained model from disk.
        
        Args:
            filename: Name of the file to load (default: 'svm_model.joblib')
        """
        filename = filename or 'svm_model.joblib'
        filepath = self.models_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.best_model = self.model
        self.training_history = model_data.get('training_history', {})
        
        print(f"Model loaded from: {filepath}")
    
    def print_kernel_comparison(self) -> None:
        """Print comparison of different kernel performances."""
        comparison = self._get_kernel_comparison()
        
        if not comparison.empty:
            print("\n" + "="*60)
            print("Kernel Performance Comparison")
            print("="*60 + "\n")
            print(comparison.to_string(index=False))
            print()
        else:
            print("No kernel comparison data available.")


if __name__ == "__main__":
    # Test the SVM model
    print("Testing SVM Model with Kernel Optimization")
    print("="*60)
    
    from ..data.data_loader import DataLoader
    from ..data.preprocessor import DataPreprocessor
    
    # Load and preprocess data
    loader = DataLoader()
    X, y = loader.load_all_modalities()
    
    preprocessor = DataPreprocessor()
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.preprocess_pipeline(X, y)
    
    # Train model
    svm_model = SVMModel()
    svm_model.train(X_train, y_train, X_val, y_val, search_method='grid')
    
    # Print kernel comparison
    svm_model.print_kernel_comparison()
    
    # Evaluate on test set
    test_metrics = svm_model.evaluate(X_test, y_test)
    
    # Save model
    svm_model.save_model()
    
    print("\nSVM Model Training Complete!")

