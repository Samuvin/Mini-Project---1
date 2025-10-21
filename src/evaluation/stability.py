"""Model stability analysis and statistical testing for Parkinson's Disease detection."""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import learning_curve, validation_curve
from pathlib import Path

from ..utils.config import Config


class StabilityAnalyzer:
    """Analyze model stability and statistical significance."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the stability analyzer.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        self.val_config = self.config.config.get('validation', {})
        
        # Set plotting style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def calculate_cv_statistics(
        self,
        cv_scores: np.ndarray,
        metric_name: str = 'Score'
    ) -> Dict[str, float]:
        """
        Calculate statistics for cross-validation scores.
        
        Args:
            cv_scores: Array of CV scores
            metric_name: Name of the metric
            
        Returns:
            Dictionary with statistics
        """
        stats_dict = {
            'mean': np.mean(cv_scores),
            'std': np.std(cv_scores),
            'min': np.min(cv_scores),
            'max': np.max(cv_scores),
            'median': np.median(cv_scores),
            'cv_coefficient': np.std(cv_scores) / np.mean(cv_scores) if np.mean(cv_scores) != 0 else 0,
            'variance': np.var(cv_scores)
        }
        
        return stats_dict
    
    def calculate_confidence_interval(
        self,
        scores: np.ndarray,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval using bootstrap.
        
        Args:
            scores: Array of scores
            confidence: Confidence level (default 0.95 for 95% CI)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        n_bootstrap = self.val_config.get('bootstrap_samples', 1000)
        
        bootstrap_means = []
        n = len(scores)
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            sample = np.random.choice(scores, size=n, replace=True)
            bootstrap_means.append(np.mean(sample))
        
        bootstrap_means = np.array(bootstrap_means)
        
        # Calculate confidence interval
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_means, (alpha/2) * 100)
        upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)
        
        return lower, upper
    
    def perform_paired_ttest(
        self,
        scores_a: np.ndarray,
        scores_b: np.ndarray,
        model_a_name: str = 'Model A',
        model_b_name: str = 'Model B'
    ) -> Dict[str, any]:
        """
        Perform paired t-test to compare two models.
        
        Args:
            scores_a: CV scores for model A
            scores_b: CV scores for model B
            model_a_name: Name of model A
            model_b_name: Name of model B
            
        Returns:
            Dictionary with test results
        """
        # Perform paired t-test
        t_statistic, p_value = stats.ttest_rel(scores_a, scores_b)
        
        # Calculate effect size (Cohen's d)
        diff = scores_a - scores_b
        cohens_d = np.mean(diff) / np.std(diff) if np.std(diff) != 0 else 0
        
        # Determine significance
        is_significant = p_value < 0.05
        
        results = {
            'model_a': model_a_name,
            'model_b': model_b_name,
            'mean_a': np.mean(scores_a),
            'mean_b': np.mean(scores_b),
            'std_a': np.std(scores_a),
            'std_b': np.std(scores_b),
            't_statistic': t_statistic,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'is_significant': is_significant,
            'winner': model_a_name if np.mean(scores_a) > np.mean(scores_b) else model_b_name
        }
        
        return results
    
    def print_stability_report(
        self,
        cv_scores: np.ndarray,
        model_name: str,
        metric_name: str = 'ROC-AUC'
    ) -> None:
        """
        Print a stability report for a model.
        
        Args:
            cv_scores: Array of CV scores
            model_name: Name of the model
            metric_name: Name of the metric
        """
        stats = self.calculate_cv_statistics(cv_scores, metric_name)
        ci_lower, ci_upper = self.calculate_confidence_interval(cv_scores)
        
        print(f"\n{'='*60}")
        print(f"Stability Report: {model_name}")
        print(f"{'='*60}")
        print(f"Metric: {metric_name}")
        print(f"\nCross-Validation Statistics:")
        print(f"  Mean:            {stats['mean']:.4f}")
        print(f"  Std Dev:         {stats['std']:.4f}")
        print(f"  Min:             {stats['min']:.4f}")
        print(f"  Max:             {stats['max']:.4f}")
        print(f"  Median:          {stats['median']:.4f}")
        print(f"  CV Coefficient:  {stats['cv_coefficient']:.4f}")
        print(f"  Variance:        {stats['variance']:.6f}")
        print(f"\n95% Confidence Interval:")
        print(f"  [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"{'='*60}\n")
    
    def compare_models_statistically(
        self,
        model_scores: Dict[str, np.ndarray],
        metric_name: str = 'ROC-AUC'
    ) -> pd.DataFrame:
        """
        Compare multiple models using statistical tests.
        
        Args:
            model_scores: Dictionary mapping model names to CV scores
            metric_name: Name of the metric
            
        Returns:
            DataFrame with pairwise comparison results
        """
        print(f"\n{'='*60}")
        print(f"Statistical Model Comparison - {metric_name}")
        print(f"{'='*60}\n")
        
        model_names = list(model_scores.keys())
        results = []
        
        # Pairwise comparisons
        for i, model_a in enumerate(model_names):
            for model_b in model_names[i+1:]:
                test_result = self.perform_paired_ttest(
                    model_scores[model_a],
                    model_scores[model_b],
                    model_a,
                    model_b
                )
                results.append(test_result)
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        print(df.to_string(index=False))
        print(f"\n{'='*60}\n")
        
        return df
    
    def plot_learning_curves(
        self,
        estimator: any,
        X: np.ndarray,
        y: np.ndarray,
        model_name: str = 'Model',
        cv: int = 5,
        save_path: Optional[Path] = None
    ) -> None:
        """
        Plot learning curves (sample size vs performance).
        
        Args:
            estimator: Model estimator
            X: Features
            y: Labels
            model_name: Name of the model
            cv: Number of CV folds
            save_path: Path to save plot
        """
        train_sizes = np.linspace(0.1, 1.0, 10)
        
        train_sizes_abs, train_scores, val_scores = learning_curve(
            estimator, X, y,
            train_sizes=train_sizes,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        plt.figure(figsize=(10, 6))
        
        plt.plot(train_sizes_abs, train_mean, 'o-', color='blue', label='Training Score')
        plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std,
                        alpha=0.2, color='blue')
        
        plt.plot(train_sizes_abs, val_mean, 'o-', color='red', label='Validation Score')
        plt.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std,
                        alpha=0.2, color='red')
        
        plt.xlabel('Training Set Size', fontsize=12)
        plt.ylabel('ROC-AUC Score', fontsize=12)
        plt.title(f'Learning Curves - {model_name}', fontsize=14, fontweight='bold')
        plt.legend(loc='best')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Learning curves saved to: {save_path}")
        
        plt.close()
    
    def plot_validation_curves(
        self,
        estimator: any,
        X: np.ndarray,
        y: np.ndarray,
        param_name: str,
        param_range: List,
        model_name: str = 'Model',
        cv: int = 5,
        save_path: Optional[Path] = None
    ) -> None:
        """
        Plot validation curves (hyperparameter vs performance).
        
        Args:
            estimator: Model estimator
            X: Features
            y: Labels
            param_name: Name of hyperparameter
            param_range: Range of hyperparameter values
            model_name: Name of the model
            cv: Number of CV folds
            save_path: Path to save plot
        """
        train_scores, val_scores = validation_curve(
            estimator, X, y,
            param_name=param_name,
            param_range=param_range,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        plt.figure(figsize=(10, 6))
        
        plt.plot(param_range, train_mean, 'o-', color='blue', label='Training Score')
        plt.fill_between(param_range, train_mean - train_std, train_mean + train_std,
                        alpha=0.2, color='blue')
        
        plt.plot(param_range, val_mean, 'o-', color='red', label='Validation Score')
        plt.fill_between(param_range, val_mean - val_std, val_mean + val_std,
                        alpha=0.2, color='red')
        
        plt.xlabel(param_name, fontsize=12)
        plt.ylabel('ROC-AUC Score', fontsize=12)
        plt.title(f'Validation Curve - {model_name}', fontsize=14, fontweight='bold')
        plt.legend(loc='best')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Validation curves saved to: {save_path}")
        
        plt.close()
    
    def analyze_bias_variance(
        self,
        train_scores: np.ndarray,
        val_scores: np.ndarray
    ) -> Dict[str, str]:
        """
        Analyze bias-variance tradeoff.
        
        Args:
            train_scores: Training scores across CV folds
            val_scores: Validation scores across CV folds
            
        Returns:
            Dictionary with bias-variance analysis
        """
        train_mean = np.mean(train_scores)
        val_mean = np.mean(val_scores)
        gap = train_mean - val_mean
        
        analysis = {}
        
        # Check for overfitting (high variance)
        if gap > 0.1:
            analysis['diagnosis'] = 'High Variance (Overfitting)'
            analysis['recommendation'] = 'Consider: regularization, reduce complexity, more data, or dropout'
        # Check for underfitting (high bias)
        elif train_mean < 0.7:
            analysis['diagnosis'] = 'High Bias (Underfitting)'
            analysis['recommendation'] = 'Consider: increase model complexity, add features, reduce regularization'
        else:
            analysis['diagnosis'] = 'Good Fit'
            analysis['recommendation'] = 'Model appears well-balanced'
        
        analysis['train_mean'] = train_mean
        analysis['val_mean'] = val_mean
        analysis['gap'] = gap
        
        return analysis
    
    def create_stability_summary(
        self,
        model_cv_results: Dict[str, Dict[str, np.ndarray]],
        save_path: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        Create a comprehensive stability summary for multiple models.
        
        Args:
            model_cv_results: Dict mapping model names to their CV results
            save_path: Path to save summary
            
        Returns:
            DataFrame with stability summary
        """
        summary_data = []
        
        for model_name, cv_scores in model_cv_results.items():
            stats = self.calculate_cv_statistics(cv_scores, 'ROC-AUC')
            ci_lower, ci_upper = self.calculate_confidence_interval(cv_scores)
            
            summary_data.append({
                'Model': model_name,
                'Mean': stats['mean'],
                'Std': stats['std'],
                'Min': stats['min'],
                'Max': stats['max'],
                'CV_Coef': stats['cv_coefficient'],
                'CI_Lower': ci_lower,
                'CI_Upper': ci_upper
            })
        
        df = pd.DataFrame(summary_data)
        df = df.sort_values('Mean', ascending=False)
        
        if save_path:
            df.to_csv(save_path, index=False)
            print(f"Stability summary saved to: {save_path}")
        
        return df


if __name__ == "__main__":
    # Test stability analyzer
    print("Testing Stability Analyzer")
    print("="*60)
    
    # Create synthetic CV scores
    np.random.seed(42)
    model_a_scores = np.random.normal(0.85, 0.05, 5)
    model_b_scores = np.random.normal(0.82, 0.08, 5)
    
    analyzer = StabilityAnalyzer()
    
    # Test stability report
    analyzer.print_stability_report(model_a_scores, 'Model A', 'ROC-AUC')
    
    # Test statistical comparison
    model_scores = {
        'Model A': model_a_scores,
        'Model B': model_b_scores
    }
    comparison_df = analyzer.compare_models_statistically(model_scores)
    
    # Test stability summary
    summary_df = analyzer.create_stability_summary(model_scores)
    print("\nStability Summary:")
    print(summary_df.to_string(index=False))
    
    print("\nStability Analyzer Test Complete!")

