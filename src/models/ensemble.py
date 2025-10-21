"""Ensemble models combining multiple classifiers for Parkinson's Disease detection."""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
import joblib
from pathlib import Path

from ..utils.config import Config, get_models_dir, ensure_dir_exists
from ..evaluation.metrics import ModelEvaluator


class EnsembleModel:
    """Ensemble model using voting or stacking strategies."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the Ensemble model.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        self.ensemble_config = self.config.config.get('models', {}).get('ensemble', {})
        self.training_config = self.config.get_training_config()
        
        self.model = None
        self.base_models = []
        self.training_history = {}
        
        self.models_dir = get_models_dir()
        ensure_dir_exists(self.models_dir)
    
    def create_voting_ensemble(
        self,
        estimators: List[Tuple[str, any]],
        voting: str = 'soft',
        weights: Optional[List[float]] = None
    ) -> VotingClassifier:
        """
        Create a voting ensemble classifier.
        
        Args:
            estimators: List of (name, estimator) tuples
            voting: Voting strategy ('soft' or 'hard')
            weights: Weights for each estimator
            
        Returns:
            VotingClassifier instance
        """
        n_jobs = self.training_config.get('n_jobs', -1)
        
        ensemble = VotingClassifier(
            estimators=estimators,
            voting=voting,
            weights=weights,
            n_jobs=n_jobs
        )
        
        return ensemble
    
    def create_stacking_ensemble(
        self,
        estimators: List[Tuple[str, any]],
        final_estimator: Optional[any] = None,
        cv: int = 5
    ) -> StackingClassifier:
        """
        Create a stacking ensemble classifier.
        
        Args:
            estimators: List of (name, estimator) tuples for base models
            final_estimator: Meta-learner (if None, uses LogisticRegression)
            cv: Number of cross-validation folds
            
        Returns:
            StackingClassifier instance
        """
        if final_estimator is None:
            final_estimator = LogisticRegression(random_state=self.config.random_state)
        
        n_jobs = self.training_config.get('n_jobs', -1)
        
        ensemble = StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=cv,
            n_jobs=n_jobs
        )
        
        return ensemble
    
    def train_voting_ensemble(
        self,
        estimators: List[Tuple[str, any]],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        weights: Optional[List[float]] = None
    ) -> Dict[str, any]:
        """
        Train a voting ensemble.
        
        Args:
            estimators: List of (name, estimator) tuples
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            weights: Weights for each estimator
            
        Returns:
            Training history dictionary
        """
        print("\n" + "="*60)
        print("Training Voting Ensemble")
        print("="*60 + "\n")
        
        voting_strategy = self.ensemble_config.get('voting_strategy', 'soft')
        
        print(f"Base models: {[name for name, _ in estimators]}")
        print(f"Voting strategy: {voting_strategy}")
        print(f"Weights: {weights if weights else 'Equal'}\n")
        
        # Create ensemble
        self.model = self.create_voting_ensemble(
            estimators=estimators,
            voting=voting_strategy,
            weights=weights
        )
        
        # Train ensemble
        import time
        start_time = time.time()
        self.model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        print(f"Training time: {training_time:.2f} seconds\n")
        
        # Evaluate on training set
        evaluator = ModelEvaluator(self.config)
        train_pred = self.model.predict(X_train)
        train_pred_proba = self.model.predict_proba(X_train)[:, 1]
        train_metrics = evaluator.calculate_metrics(y_train, train_pred, train_pred_proba)
        
        print("Training Set Performance:")
        evaluator.print_metrics(train_metrics)
        
        # Evaluate on validation set if provided
        val_metrics = None
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_pred_proba = self.model.predict_proba(X_val)[:, 1]
            val_metrics = evaluator.calculate_metrics(y_val, val_pred, val_pred_proba)
            
            print("\nValidation Set Performance:")
            evaluator.print_metrics(val_metrics)
        
        self.training_history = {
            'ensemble_type': 'voting',
            'voting_strategy': voting_strategy,
            'base_models': [name for name, _ in estimators],
            'weights': weights,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'training_time': training_time
        }
        
        print("\n" + "="*60)
        print("Voting Ensemble Training Complete")
        print("="*60 + "\n")
        
        return self.training_history
    
    def train_stacking_ensemble(
        self,
        estimators: List[Tuple[str, any]],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        final_estimator: Optional[any] = None
    ) -> Dict[str, any]:
        """
        Train a stacking ensemble.
        
        Args:
            estimators: List of (name, estimator) tuples
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            final_estimator: Meta-learner
            
        Returns:
            Training history dictionary
        """
        print("\n" + "="*60)
        print("Training Stacking Ensemble")
        print("="*60 + "\n")
        
        print(f"Base models: {[name for name, _ in estimators]}")
        print(f"Meta-learner: {type(final_estimator).__name__ if final_estimator else 'LogisticRegression'}\n")
        
        # Create ensemble
        cv_folds = self.training_config.get('cv_folds', 5)
        self.model = self.create_stacking_ensemble(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=cv_folds
        )
        
        # Train ensemble
        import time
        start_time = time.time()
        self.model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        print(f"Training time: {training_time:.2f} seconds\n")
        
        # Evaluate on training set
        evaluator = ModelEvaluator(self.config)
        train_pred = self.model.predict(X_train)
        train_pred_proba = self.model.predict_proba(X_train)[:, 1]
        train_metrics = evaluator.calculate_metrics(y_train, train_pred, train_pred_proba)
        
        print("Training Set Performance:")
        evaluator.print_metrics(train_metrics)
        
        # Evaluate on validation set if provided
        val_metrics = None
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_pred_proba = self.model.predict_proba(X_val)[:, 1]
            val_metrics = evaluator.calculate_metrics(y_val, val_pred, val_pred_proba)
            
            print("\nValidation Set Performance:")
            evaluator.print_metrics(val_metrics)
        
        self.training_history = {
            'ensemble_type': 'stacking',
            'base_models': [name for name, _ in estimators],
            'meta_learner': type(final_estimator).__name__ if final_estimator else 'LogisticRegression',
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'training_time': training_time
        }
        
        print("\n" + "="*60)
        print("Stacking Ensemble Training Complete")
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
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        ensemble_name: str = 'Ensemble'
    ) -> Dict[str, float]:
        """Evaluate the ensemble on test data."""
        print("\n" + "="*60)
        print(f"Evaluating {ensemble_name} on Test Set")
        print("="*60 + "\n")
        
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)[:, 1]
        
        evaluator = ModelEvaluator(self.config)
        test_metrics = evaluator.calculate_metrics(y_test, y_pred, y_pred_proba)
        evaluator.print_metrics(test_metrics)
        
        # Generate visualizations
        evaluator.plot_confusion_matrix(
            y_test, y_pred,
            save_path=self.models_dir / f'{ensemble_name.lower().replace(" ", "_")}_confusion_matrix.png'
        )
        
        evaluator.plot_roc_curve(
            y_test, y_pred_proba,
            model_name=ensemble_name,
            save_path=self.models_dir / f'{ensemble_name.lower().replace(" ", "_")}_roc_curve.png'
        )
        
        return test_metrics
    
    def calculate_cv_based_weights(
        self,
        model_cv_scores: Dict[str, float]
    ) -> List[float]:
        """
        Calculate weights based on cross-validation scores.
        
        Args:
            model_cv_scores: Dictionary mapping model names to CV scores
            
        Returns:
            List of weights normalized to sum to 1
        """
        scores = np.array(list(model_cv_scores.values()))
        
        # Convert scores to weights (higher score = higher weight)
        # Use softmax-like normalization
        weights = np.exp(scores) / np.sum(np.exp(scores))
        
        print("\nCalculated CV-based weights:")
        for name, weight in zip(model_cv_scores.keys(), weights):
            print(f"  {name}: {weight:.4f}")
        
        return weights.tolist()
    
    def save_model(self, filename: Optional[str] = None) -> None:
        """Save the trained ensemble model."""
        if self.model is None:
            raise ValueError("No model to save.")
        
        filename = filename or 'ensemble_model.joblib'
        filepath = self.models_dir / filename
        
        model_data = {
            'model': self.model,
            'training_history': self.training_history,
            'config': self.config.config
        }
        
        joblib.dump(model_data, filepath)
        print(f"Ensemble model saved to: {filepath}")
    
    def load_model(self, filename: Optional[str] = None) -> None:
        """Load a trained ensemble model."""
        filename = filename or 'ensemble_model.joblib'
        filepath = self.models_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.training_history = model_data.get('training_history', {})
        
        print(f"Ensemble model loaded from: {filepath}")


def create_ensemble_from_trained_models(
    trained_models: Dict[str, any],
    ensemble_type: str = 'voting',
    config: Optional[Config] = None
) -> EnsembleModel:
    """
    Create an ensemble from already-trained models.
    
    Args:
        trained_models: Dictionary mapping model names to trained model objects
        ensemble_type: Type of ensemble ('voting' or 'stacking')
        config: Configuration object
        
    Returns:
        EnsembleModel instance
    """
    ensemble = EnsembleModel(config)
    
    # Prepare estimators list
    estimators = [(name, model.model) for name, model in trained_models.items()]
    
    if ensemble_type == 'voting':
        # Calculate weights based on validation performance if available
        weights = None
        if config and config.config.get('models', {}).get('ensemble', {}).get('weights') == 'cv_based':
            cv_scores = {}
            for name, model in trained_models.items():
                if hasattr(model, 'training_history') and 'best_score' in model.training_history:
                    cv_scores[name] = model.training_history['best_score']
            
            if cv_scores:
                weights = ensemble.calculate_cv_based_weights(cv_scores)
        
        ensemble.model = ensemble.create_voting_ensemble(estimators, weights=weights)
    
    elif ensemble_type == 'stacking':
        ensemble.model = ensemble.create_stacking_ensemble(estimators)
    
    else:
        raise ValueError(f"Unknown ensemble type: {ensemble_type}")
    
    return ensemble

