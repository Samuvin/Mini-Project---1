"""Logistic Regression baseline model for Parkinson's Disease prediction."""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import joblib
from pathlib import Path

from ..utils.config import Config, get_models_dir, ensure_dir_exists
from ..evaluation.metrics import ModelEvaluator


class LogisticRegressionModel:
    """Logistic Regression baseline model with hyperparameter tuning."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the Logistic Regression model.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        self.model_config = self.config.get_model_config('logistic_regression')
        self.training_config = self.config.get_training_config()
        
        self.model = None
        self.best_model = None
        self.grid_search = None
        self.training_history = {}
        
        self.models_dir = get_models_dir()
        ensure_dir_exists(self.models_dir)
    
    def create_model(
        self,
        C: float = 1.0,
        penalty: str = 'l2',
        solver: str = 'liblinear',
        class_weight: str = 'balanced'
    ) -> LogisticRegression:
        """
        Create a Logistic Regression model with specified parameters.
        
        Args:
            C: Inverse regularization strength
            penalty: Regularization penalty ('l1' or 'l2')
            solver: Optimization algorithm
            class_weight: Class weight strategy
            
        Returns:
            LogisticRegression model instance
        """
        max_iter = self.model_config.get('max_iter', 1000)
        random_state = self.config.random_state
        
        model = LogisticRegression(
            C=C,
            penalty=penalty,
            solver=solver,
            class_weight=class_weight,
            max_iter=max_iter,
            random_state=random_state
        )
        
        return model
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, any]:
        """
        Train the Logistic Regression model with hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Dictionary containing training results and metrics
        """
        print("\n" + "="*60)
        print("Training Logistic Regression Baseline Model")
        print("="*60 + "\n")
        
        # Prepare parameter grid for GridSearchCV
        param_grid = {
            'C': self.model_config.get('C', [0.001, 0.01, 0.1, 1, 10, 100]),
            'penalty': self.model_config.get('penalty', ['l1', 'l2']),
            'solver': self.model_config.get('solver', ['liblinear', 'saga'])
        }
        
        # Create base model
        base_model = self.create_model()
        
        # Perform grid search with cross-validation
        cv_folds = self.training_config.get('cv_folds', 5)
        scoring = self.training_config.get('scoring', 'roc_auc')
        n_jobs = self.training_config.get('n_jobs', -1)
        verbose = self.training_config.get('verbose', 1)
        
        print(f"Performing GridSearchCV with {cv_folds}-fold cross-validation...")
        print(f"Parameter grid: {param_grid}")
        print(f"Scoring metric: {scoring}\n")
        
        self.grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=cv_folds,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose,
            return_train_score=True
        )
        
        # Fit grid search
        self.grid_search.fit(X_train, y_train)
        
        # Get best model
        self.best_model = self.grid_search.best_estimator_
        self.model = self.best_model
        
        print("\n" + "-"*60)
        print("Best Parameters:")
        for param, value in self.grid_search.best_params_.items():
            print(f"  {param}: {value}")
        print(f"\nBest Cross-Validation Score ({scoring}): {self.grid_search.best_score_:.4f}")
        print("-"*60 + "\n")
        
        # Evaluate on training set
        train_pred = self.best_model.predict(X_train)
        train_pred_proba = self.best_model.predict_proba(X_train)[:, 1]
        
        evaluator = ModelEvaluator(self.config)
        train_metrics = evaluator.calculate_metrics(y_train, train_pred, train_pred_proba)
        
        print("Training Set Performance:")
        evaluator.print_metrics(train_metrics)
        
        # Evaluate on validation set if provided
        val_metrics = None
        if X_val is not None and y_val is not None:
            val_pred = self.best_model.predict(X_val)
            val_pred_proba = self.best_model.predict_proba(X_val)[:, 1]
            
            val_metrics = evaluator.calculate_metrics(y_val, val_pred, val_pred_proba)
            
            print("\nValidation Set Performance:")
            evaluator.print_metrics(val_metrics)
        
        # Store training history
        self.training_history = {
            'best_params': self.grid_search.best_params_,
            'best_score': self.grid_search.best_score_,
            'cv_results': self.grid_search.cv_results_,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics
        }
        
        print("\n" + "="*60)
        print("Logistic Regression Training Complete")
        print("="*60 + "\n")
        
        return self.training_history
    
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
        print("Evaluating Logistic Regression on Test Set")
        print("="*60 + "\n")
        
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)[:, 1]
        
        evaluator = ModelEvaluator(self.config)
        test_metrics = evaluator.calculate_metrics(y_test, y_pred, y_pred_proba)
        
        evaluator.print_metrics(test_metrics)
        
        # Generate visualizations
        evaluator.plot_confusion_matrix(
            y_test, y_pred,
            save_path=self.models_dir / 'lr_confusion_matrix.png'
        )
        
        evaluator.plot_roc_curve(
            y_test, y_pred_proba,
            model_name='Logistic Regression',
            save_path=self.models_dir / 'lr_roc_curve.png'
        )
        
        return test_metrics
    
    def save_model(self, filename: Optional[str] = None) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filename: Name of the file to save (default: 'logistic_regression_model.joblib')
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        filename = filename or 'logistic_regression_model.joblib'
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
            filename: Name of the file to load (default: 'logistic_regression_model.joblib')
        """
        filename = filename or 'logistic_regression_model.joblib'
        filepath = self.models_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.best_model = self.model
        self.training_history = model_data.get('training_history', {})
        
        print(f"Model loaded from: {filepath}")
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance (coefficients) from the model.
        
        Returns:
            DataFrame with feature names and their importance
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        coefficients = self.model.coef_[0]
        importance = np.abs(coefficients)
        
        feature_importance = pd.DataFrame({
            'feature': [f'feature_{i}' for i in range(len(importance))],
            'importance': importance,
            'coefficient': coefficients
        })
        
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        return feature_importance


if __name__ == "__main__":
    # Test the Logistic Regression model
    print("Testing Logistic Regression Model")
    print("="*60)
    
    from ..data.data_loader import DataLoader
    from ..data.preprocessor import DataPreprocessor
    
    # Load and preprocess data
    loader = DataLoader()
    X, y = loader.load_all_modalities()
    
    preprocessor = DataPreprocessor()
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.preprocess_pipeline(X, y)
    
    # Train model
    lr_model = LogisticRegressionModel()
    lr_model.train(X_train, y_train, X_val, y_val)
    
    # Evaluate on test set
    test_metrics = lr_model.evaluate(X_test, y_test)
    
    # Save model
    lr_model.save_model()
    
    print("\nLogistic Regression Baseline Complete!")

