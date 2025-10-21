"""Random Forest model for Parkinson's Disease detection."""

import numpy as np
import pandas as pd
from typing import Optional, Dict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import joblib
from pathlib import Path
import time

from ..utils.config import Config, get_models_dir, ensure_dir_exists
from ..evaluation.metrics import ModelEvaluator


class RandomForestModel:
    """Random Forest model with hyperparameter tuning and OOB error estimation."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the Random Forest model.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        self.model_config = self.config.get_model_config('random_forest')
        self.training_config = self.config.get_training_config()
        
        self.model = None
        self.best_model = None
        self.grid_search = None
        self.training_history = {}
        
        self.models_dir = get_models_dir()
        ensure_dir_exists(self.models_dir)
    
    def create_model(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str = 'sqrt',
        class_weight: str = 'balanced',
        oob_score: bool = True
    ) -> RandomForestClassifier:
        """
        Create a Random Forest model with specified parameters.
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees
            min_samples_split: Minimum samples required to split a node
            min_samples_leaf: Minimum samples required at leaf node
            max_features: Number of features to consider for best split
            class_weight: Class weight strategy
            oob_score: Whether to use out-of-bag samples to estimate error
            
        Returns:
            RandomForestClassifier instance
        """
        random_state = self.config.random_state
        n_jobs = self.training_config.get('n_jobs', -1)
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight=class_weight,
            oob_score=oob_score,
            random_state=random_state,
            n_jobs=n_jobs
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
        Train the Random Forest model with hyperparameter tuning.
        
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
        print("Training Random Forest Model")
        print("="*60 + "\n")
        
        # Prepare parameter grid for hyperparameter search
        param_grid = {
            'n_estimators': self.model_config.get('n_estimators', [100, 200, 300]),
            'max_depth': self.model_config.get('max_depth', [10, 20, 30, None]),
            'min_samples_split': self.model_config.get('min_samples_split', [2, 5, 10]),
            'min_samples_leaf': self.model_config.get('min_samples_leaf', [1, 2, 4]),
            'max_features': self.model_config.get('max_features', ['sqrt', 'log2'])
        }
        
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
        
        # Print OOB score if available
        if hasattr(self.best_model, 'oob_score_'):
            print(f"Out-of-Bag Score: {self.best_model.oob_score_:.4f}")
        
        print(f"Training time: {training_time:.2f} seconds")
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
            'val_metrics': val_metrics,
            'training_time': training_time,
            'oob_score': self.best_model.oob_score_ if hasattr(self.best_model, 'oob_score_') else None
        }
        
        print("\n" + "="*60)
        print("Random Forest Training Complete")
        print("="*60 + "\n")
        
        return self.training_history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict_proba(X)
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate the model on test data."""
        print("\n" + "="*60)
        print("Evaluating Random Forest on Test Set")
        print("="*60 + "\n")
        
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)[:, 1]
        
        evaluator = ModelEvaluator(self.config)
        test_metrics = evaluator.calculate_metrics(y_test, y_pred, y_pred_proba)
        
        evaluator.print_metrics(test_metrics)
        
        # Generate visualizations
        evaluator.plot_confusion_matrix(
            y_test, y_pred,
            save_path=self.models_dir / 'rf_confusion_matrix.png'
        )
        
        evaluator.plot_roc_curve(
            y_test, y_pred_proba,
            model_name='Random Forest',
            save_path=self.models_dir / 'rf_roc_curve.png'
        )
        
        return test_metrics
    
    def get_feature_importance(self, feature_names: Optional[list] = None) -> pd.DataFrame:
        """
        Get feature importance from the model.
        
        Args:
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature names and their importance
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        importances = self.model.feature_importances_
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importances))]
        
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
        
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        return feature_importance
    
    def save_model(self, filename: Optional[str] = None) -> None:
        """Save the trained model to disk."""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        filename = filename or 'random_forest_model.joblib'
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
        """Load a trained model from disk."""
        filename = filename or 'random_forest_model.joblib'
        filepath = self.models_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.best_model = self.model
        self.training_history = model_data.get('training_history', {})
        
        print(f"Model loaded from: {filepath}")

