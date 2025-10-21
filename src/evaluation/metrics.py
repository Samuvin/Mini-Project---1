"""Model evaluation metrics and visualization utilities."""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve
)
from pathlib import Path

from ..utils.config import Config


class ModelEvaluator:
    """Evaluate machine learning models with comprehensive metrics and visualizations."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the model evaluator.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        self.eval_config = self.config.get_evaluation_config()
        
        # Set plotting style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
        }
        
        # Add ROC-AUC if probabilities are provided
        if y_pred_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            except ValueError:
                metrics['roc_auc'] = 0.0
        
        # Calculate confusion matrix values
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)
        
        # Calculate specificity
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # Calculate sensitivity (same as recall)
        metrics['sensitivity'] = metrics['recall']
        
        return metrics
    
    def print_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Print metrics in a formatted way.
        
        Args:
            metrics: Dictionary of metrics
        """
        print("Classification Metrics:")
        print(f"  Accuracy:    {metrics['accuracy']:.4f}")
        print(f"  Precision:   {metrics['precision']:.4f}")
        print(f"  Recall:      {metrics['recall']:.4f}")
        print(f"  F1-Score:    {metrics['f1_score']:.4f}")
        
        if 'roc_auc' in metrics:
            print(f"  ROC-AUC:     {metrics['roc_auc']:.4f}")
        
        print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")
        
        print("\nConfusion Matrix Values:")
        print(f"  True Positives:  {metrics['true_positives']}")
        print(f"  True Negatives:  {metrics['true_negatives']}")
        print(f"  False Positives: {metrics['false_positives']}")
        print(f"  False Negatives: {metrics['false_negatives']}")
    
    def get_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_names: Optional[List[str]] = None
    ) -> str:
        """
        Generate a classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            target_names: Names of target classes
            
        Returns:
            Classification report as string
        """
        if target_names is None:
            target_names = ['Healthy', 'Parkinson\'s']
        
        return classification_report(y_true, y_pred, target_names=target_names)
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[List[str]] = None,
        save_path: Optional[Path] = None,
        show: bool = False
    ) -> None:
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Class labels
            save_path: Path to save the plot
            show: Whether to display the plot
        """
        if labels is None:
            labels = ['Healthy', 'Parkinson\'s']
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels,
            cbar_kws={'label': 'Count'}
        )
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        model_name: str = 'Model',
        save_path: Optional[Path] = None,
        show: bool = False
    ) -> None:
        """
        Plot ROC curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            model_name: Name of the model
            save_path: Path to save the plot
            show: Whether to display the plot
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'{model_name} (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        model_name: str = 'Model',
        save_path: Optional[Path] = None,
        show: bool = False
    ) -> None:
        """
        Plot Precision-Recall curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            model_name: Name of the model
            save_path: Path to save the plot
            show: Whether to display the plot
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2,
                 label=f'{model_name} (AUC = {pr_auc:.4f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=16, fontweight='bold')
        plt.legend(loc='lower left', fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Precision-Recall curve saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def compare_models(
        self,
        models_metrics: Dict[str, Dict[str, float]],
        save_path: Optional[Path] = None,
        show: bool = False
    ) -> pd.DataFrame:
        """
        Compare metrics across multiple models.
        
        Args:
            models_metrics: Dictionary mapping model names to their metrics
            save_path: Path to save the comparison plot
            show: Whether to display the plot
            
        Returns:
            DataFrame with model comparison
        """
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(models_metrics).T
        
        # Select key metrics for visualization
        key_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        available_metrics = [m for m in key_metrics if m in comparison_df.columns]
        
        if not available_metrics:
            print("No metrics available for comparison")
            return comparison_df
        
        # Plot comparison
        fig, ax = plt.subplots(figsize=(12, 6))
        
        comparison_df[available_metrics].plot(kind='bar', ax=ax, width=0.8)
        
        ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold')
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_ylim([0, 1.1])
        ax.legend(title='Metrics', loc='lower right')
        ax.grid(axis='y', alpha=0.3)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Model comparison saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return comparison_df
    
    def plot_learning_curve(
        self,
        train_scores: List[float],
        val_scores: List[float],
        metric_name: str = 'Score',
        save_path: Optional[Path] = None,
        show: bool = False
    ) -> None:
        """
        Plot learning curves.
        
        Args:
            train_scores: Training scores over iterations
            val_scores: Validation scores over iterations
            metric_name: Name of the metric being plotted
            save_path: Path to save the plot
            show: Whether to display the plot
        """
        epochs = range(1, len(train_scores) + 1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_scores, 'b-', label='Training', linewidth=2)
        plt.plot(epochs, val_scores, 'r-', label='Validation', linewidth=2)
        
        plt.title(f'Learning Curve - {metric_name}', fontsize=16, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel(metric_name, fontsize=12)
        plt.legend(loc='best', fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Learning curve saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def generate_evaluation_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray],
        model_name: str,
        output_dir: Path
    ) -> Dict[str, any]:
        """
        Generate a comprehensive evaluation report with all metrics and plots.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            model_name: Name of the model
            output_dir: Directory to save outputs
            
        Returns:
            Dictionary with all evaluation results
        """
        from ..utils.config import ensure_dir_exists
        ensure_dir_exists(output_dir)
        
        print(f"\nGenerating evaluation report for {model_name}...")
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_true, y_pred, y_pred_proba)
        
        # Print metrics
        print(f"\n{model_name} Performance:")
        self.print_metrics(metrics)
        
        # Generate plots
        self.plot_confusion_matrix(
            y_true, y_pred,
            save_path=output_dir / f'{model_name}_confusion_matrix.png'
        )
        
        if y_pred_proba is not None:
            self.plot_roc_curve(
                y_true, y_pred_proba,
                model_name=model_name,
                save_path=output_dir / f'{model_name}_roc_curve.png'
            )
            
            self.plot_precision_recall_curve(
                y_true, y_pred_proba,
                model_name=model_name,
                save_path=output_dir / f'{model_name}_pr_curve.png'
            )
        
        # Get classification report
        report = self.get_classification_report(y_true, y_pred)
        
        print(f"\nClassification Report:\n{report}")
        
        # Save report to file
        report_file = output_dir / f'{model_name}_report.txt'
        with open(report_file, 'w') as f:
            f.write(f"Evaluation Report: {model_name}\n")
            f.write("="*60 + "\n\n")
            f.write("Metrics:\n")
            for key, value in metrics.items():
                f.write(f"  {key}: {value}\n")
            f.write("\nClassification Report:\n")
            f.write(report)
        
        print(f"\nEvaluation report saved to: {report_file}")
        
        return {
            'metrics': metrics,
            'classification_report': report
        }


if __name__ == "__main__":
    # Test the evaluator
    print("Testing Model Evaluator")
    print("="*60)
    
    # Create synthetic test data
    np.random.seed(42)
    y_true = np.array([0, 1, 1, 0, 1, 1, 0, 0, 1, 0] * 10)
    y_pred = y_true.copy()
    # Add some errors
    y_pred[::10] = 1 - y_pred[::10]
    y_pred_proba = np.random.rand(len(y_true))
    y_pred_proba[y_true == 1] += 0.3
    y_pred_proba = np.clip(y_pred_proba, 0, 1)
    
    evaluator = ModelEvaluator()
    
    # Calculate and print metrics
    metrics = evaluator.calculate_metrics(y_true, y_pred, y_pred_proba)
    evaluator.print_metrics(metrics)
    
    # Generate plots
    evaluator.plot_confusion_matrix(y_true, y_pred, show=True)
    evaluator.plot_roc_curve(y_true, y_pred_proba, show=True)
    
    print("\nEvaluator test complete!")

