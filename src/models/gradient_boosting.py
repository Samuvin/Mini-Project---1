"""Gradient Boosting models (XGBoost and LightGBM) for Parkinson's Disease detection."""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Literal
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import joblib
from pathlib import Path
import time

from ..utils.config import Config, get_models_dir, ensure_dir_exists
from ..evaluation.metrics import ModelEvaluator


class XGBoostModel:
    """XGBoost model with hyperparameter tuning and early stopping."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the XGBoost model.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        self.model_config = self.config.config.get('models', {}).get('gradient_boosting', {}).get('xgboost', {})
        self.training_config = self.config.get_training_config()
        
        self.model = None
        self.best_model = None
        self.grid_search = None
        self.training_history = {}
        
        self.models_dir = get_models_dir()
        ensure_dir_exists(self.models_dir)
    
    def create_model(
        self,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        max_depth: int = 5,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 1,
        gamma: float = 0,
        **kwargs
    ) -> xgb.XGBClassifier:
        """
        Create an XGBoost model with specified parameters.
        
        Args:
            learning_rate: Boosting learning rate
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of columns
            min_child_weight: Minimum sum of instance weight needed in a child
            gamma: Minimum loss reduction required for split
            **kwargs: Additional XGBoost parameters
            
        Returns:
            XGBClassifier instance
        """
        random_state = self.config.random_state
        n_jobs = self.training_config.get('n_jobs', -1)
        
        model = xgb.XGBClassifier(
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            gamma=gamma,
            random_state=random_state,
            n_jobs=n_jobs,
            eval_metric='logloss',
            **kwargs
        )
        
        return model
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        search_method: str = 'grid',
        early_stopping_rounds: int = 50
    ) -> Dict[str, any]:
        """
        Train the XGBoost model with hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            search_method: Search method ('grid' or 'random')
            early_stopping_rounds: Rounds for early stopping
            
        Returns:
            Dictionary containing training results and metrics
        """
        print("\n" + "="*60)
        print("Training XGBoost Model")
        print("="*60 + "\n")
        
        # Prepare parameter grid
        param_grid = {
            'learning_rate': self.model_config.get('learning_rate', [0.01, 0.05, 0.1]),
            'n_estimators': self.model_config.get('n_estimators', [100, 200, 300]),
            'max_depth': self.model_config.get('max_depth', [3, 5, 7]),
            'subsample': self.model_config.get('subsample', [0.7, 0.8, 0.9]),
            'colsample_bytree': self.model_config.get('colsample_bytree', [0.7, 0.8, 0.9]),
            'min_child_weight': self.model_config.get('min_child_weight', [1, 3, 5]),
            'gamma': self.model_config.get('gamma', [0, 0.1, 0.2])
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
                n_iter=30,
                cv=cv_folds,
                scoring=scoring,
                n_jobs=n_jobs,
                verbose=verbose,
                random_state=self.config.random_state,
                return_train_score=True
            )
        
        self.grid_search.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        self.best_model = self.grid_search.best_estimator_
        self.model = self.best_model
        
        print("\n" + "-"*60)
        print("Best Parameters:")
        for param, value in self.grid_search.best_params_.items():
            print(f"  {param}: {value}")
        print(f"\nBest Cross-Validation Score ({scoring}): {self.grid_search.best_score_:.4f}")
        print(f"Training time: {training_time:.2f} seconds")
        print("-"*60 + "\n")
        
        # Evaluate
        evaluator = ModelEvaluator(self.config)
        train_pred = self.model.predict(X_train)
        train_pred_proba = self.model.predict_proba(X_train)[:, 1]
        train_metrics = evaluator.calculate_metrics(y_train, train_pred, train_pred_proba)
        
        print("Training Set Performance:")
        evaluator.print_metrics(train_metrics)
        
        val_metrics = None
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_pred_proba = self.model.predict_proba(X_val)[:, 1]
            val_metrics = evaluator.calculate_metrics(y_val, val_pred, val_pred_proba)
            
            print("\nValidation Set Performance:")
            evaluator.print_metrics(val_metrics)
        
        self.training_history = {
            'best_params': self.grid_search.best_params_,
            'best_score': self.grid_search.best_score_,
            'cv_results': self.grid_search.cv_results_,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'training_time': training_time
        }
        
        print("\n" + "="*60)
        print("XGBoost Training Complete")
        print("="*60 + "\n")
        
        return self.training_history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained.")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        if self.model is None:
            raise ValueError("Model not trained.")
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate on test data."""
        print("\n" + "="*60)
        print("Evaluating XGBoost on Test Set")
        print("="*60 + "\n")
        
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)[:, 1]
        
        evaluator = ModelEvaluator(self.config)
        test_metrics = evaluator.calculate_metrics(y_test, y_pred, y_pred_proba)
        evaluator.print_metrics(test_metrics)
        
        evaluator.plot_confusion_matrix(
            y_test, y_pred,
            save_path=self.models_dir / 'xgb_confusion_matrix.png'
        )
        
        evaluator.plot_roc_curve(
            y_test, y_pred_proba,
            model_name='XGBoost',
            save_path=self.models_dir / 'xgb_roc_curve.png'
        )
        
        return test_metrics
    
    def get_feature_importance(self, feature_names: Optional[list] = None) -> pd.DataFrame:
        """Get feature importance."""
        if self.model is None:
            raise ValueError("Model not trained.")
        
        importances = self.model.feature_importances_
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importances))]
        
        return pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
    
    def save_model(self, filename: Optional[str] = None) -> None:
        """Save model."""
        if self.model is None:
            raise ValueError("No model to save.")
        
        filename = filename or 'xgboost_model.joblib'
        filepath = self.models_dir / filename
        
        joblib.dump({
            'model': self.model,
            'best_params': self.grid_search.best_params_ if self.grid_search else None,
            'training_history': self.training_history,
            'config': self.config.config
        }, filepath)
        print(f"Model saved to: {filepath}")
    
    def load_model(self, filename: Optional[str] = None) -> None:
        """Load model."""
        filename = filename or 'xgboost_model.joblib'
        filepath = self.models_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.best_model = self.model
        self.training_history = model_data.get('training_history', {})
        print(f"Model loaded from: {filepath}")


class LightGBMModel:
    """LightGBM model with hyperparameter tuning and early stopping."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize LightGBM model."""
        self.config = config or Config()
        self.model_config = self.config.config.get('models', {}).get('gradient_boosting', {}).get('lightgbm', {})
        self.training_config = self.config.get_training_config()
        
        self.model = None
        self.best_model = None
        self.grid_search = None
        self.training_history = {}
        
        self.models_dir = get_models_dir()
        ensure_dir_exists(self.models_dir)
    
    def create_model(
        self,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        max_depth: int = 5,
        num_leaves: int = 31,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_samples: int = 20,
        **kwargs
    ) -> lgb.LGBMClassifier:
        """Create LightGBM model."""
        random_state = self.config.random_state
        n_jobs = self.training_config.get('n_jobs', -1)
        
        model = lgb.LGBMClassifier(
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            max_depth=max_depth,
            num_leaves=num_leaves,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_samples=min_child_samples,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=-1,
            **kwargs
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
        """Train LightGBM model."""
        print("\n" + "="*60)
        print("Training LightGBM Model")
        print("="*60 + "\n")
        
        param_grid = {
            'learning_rate': self.model_config.get('learning_rate', [0.01, 0.05, 0.1]),
            'n_estimators': self.model_config.get('n_estimators', [100, 200, 300]),
            'max_depth': self.model_config.get('max_depth', [3, 5, 7]),
            'num_leaves': self.model_config.get('num_leaves', [31, 63]),
            'subsample': self.model_config.get('subsample', [0.7, 0.8, 0.9]),
            'colsample_bytree': self.model_config.get('colsample_bytree', [0.7, 0.8, 0.9]),
            'min_child_samples': self.model_config.get('min_child_samples', [10, 20, 30])
        }
        
        base_model = self.create_model()
        
        cv_folds = self.training_config.get('cv_folds', 5)
        scoring = self.training_config.get('scoring', 'roc_auc')
        n_jobs = self.training_config.get('n_jobs', -1)
        verbose = self.training_config.get('verbose', 1)
        
        print(f"Hyperparameter search method: {search_method.upper()}")
        print(f"Cross-validation folds: {cv_folds}")
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
                n_iter=30,
                cv=cv_folds,
                scoring=scoring,
                n_jobs=n_jobs,
                verbose=verbose,
                random_state=self.config.random_state,
                return_train_score=True
            )
        
        self.grid_search.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        self.best_model = self.grid_search.best_estimator_
        self.model = self.best_model
        
        print("\n" + "-"*60)
        print("Best Parameters:")
        for param, value in self.grid_search.best_params_.items():
            print(f"  {param}: {value}")
        print(f"\nBest Cross-Validation Score ({scoring}): {self.grid_search.best_score_:.4f}")
        print(f"Training time: {training_time:.2f} seconds")
        print("-"*60 + "\n")
        
        evaluator = ModelEvaluator(self.config)
        train_pred = self.model.predict(X_train)
        train_pred_proba = self.model.predict_proba(X_train)[:, 1]
        train_metrics = evaluator.calculate_metrics(y_train, train_pred, train_pred_proba)
        
        print("Training Set Performance:")
        evaluator.print_metrics(train_metrics)
        
        val_metrics = None
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_pred_proba = self.model.predict_proba(X_val)[:, 1]
            val_metrics = evaluator.calculate_metrics(y_val, val_pred, val_pred_proba)
            
            print("\nValidation Set Performance:")
            evaluator.print_metrics(val_metrics)
        
        self.training_history = {
            'best_params': self.grid_search.best_params_,
            'best_score': self.grid_search.best_score_,
            'cv_results': self.grid_search.cv_results_,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'training_time': training_time
        }
        
        print("\n" + "="*60)
        print("LightGBM Training Complete")
        print("="*60 + "\n")
        
        return self.training_history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained.")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        if self.model is None:
            raise ValueError("Model not trained.")
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate on test data."""
        print("\n" + "="*60)
        print("Evaluating LightGBM on Test Set")
        print("="*60 + "\n")
        
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)[:, 1]
        
        evaluator = ModelEvaluator(self.config)
        test_metrics = evaluator.calculate_metrics(y_test, y_pred, y_pred_proba)
        evaluator.print_metrics(test_metrics)
        
        evaluator.plot_confusion_matrix(
            y_test, y_pred,
            save_path=self.models_dir / 'lgb_confusion_matrix.png'
        )
        
        evaluator.plot_roc_curve(
            y_test, y_pred_proba,
            model_name='LightGBM',
            save_path=self.models_dir / 'lgb_roc_curve.png'
        )
        
        return test_metrics
    
    def get_feature_importance(self, feature_names: Optional[list] = None) -> pd.DataFrame:
        """Get feature importance."""
        if self.model is None:
            raise ValueError("Model not trained.")
        
        importances = self.model.feature_importances_
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importances))]
        
        return pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
    
    def save_model(self, filename: Optional[str] = None) -> None:
        """Save model."""
        if self.model is None:
            raise ValueError("No model to save.")
        
        filename = filename or 'lightgbm_model.joblib'
        filepath = self.models_dir / filename
        
        joblib.dump({
            'model': self.model,
            'best_params': self.grid_search.best_params_ if self.grid_search else None,
            'training_history': self.training_history,
            'config': self.config.config
        }, filepath)
        print(f"Model saved to: {filepath}")
    
    def load_model(self, filename: Optional[str] = None) -> None:
        """Load model."""
        filename = filename or 'lightgbm_model.joblib'
        filepath = self.models_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.best_model = self.model
        self.training_history = model_data.get('training_history', {})
        print(f"Model loaded from: {filepath}")

