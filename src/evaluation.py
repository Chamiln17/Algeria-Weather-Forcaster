"""
Evaluation Metrics
Functions for model evaluation and performance assessment.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger(__name__)


def calculate_forecast_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = ''
) -> Dict[str, float]:
    """
    Calculate comprehensive forecast evaluation metrics.
    
    Parameters:
        y_true: Actual values
        y_pred: Predicted values
        prefix: Prefix for metric names (e.g., 'test_', 'val_')
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Mean Absolute Error
    metrics[f'{prefix}mae'] = mean_absolute_error(y_true, y_pred)
    
    # Root Mean Squared Error
    metrics[f'{prefix}rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # R² Score
    metrics[f'{prefix}r2'] = r2_score(y_true, y_pred)
    
    # Mean Absolute Percentage Error
    mask = y_true != 0
    if mask.any():
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        metrics[f'{prefix}mape'] = mape
    else:
        metrics[f'{prefix}mape'] = np.nan
    
    # Mean Error (bias)
    metrics[f'{prefix}me'] = np.mean(y_pred - y_true)
    
    # Standard deviation of errors
    metrics[f'{prefix}std_error'] = np.std(y_pred - y_true)
    
    return metrics


def calculate_directional_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    """
    Calculate directional accuracy (% of correctly predicted directions).
    
    Useful for time series: did we predict the correct direction of change?
    """
    if len(y_true) < 2:
        return np.nan
    
    true_direction = np.diff(y_true) > 0
    pred_direction = np.diff(y_pred) > 0
    
    accuracy = np.mean(true_direction == pred_direction) * 100
    return accuracy


def plot_forecast_vs_actual(
    dates: pd.DatetimeIndex,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = 'Forecast vs Actual',
    confidence_intervals: tuple = None,
    figsize: tuple = (14, 6)
) -> plt.Figure:
    """
    Plot forecast against actual values.
    
    Parameters:
        dates: Datetime index
        y_true: Actual values
        y_pred: Predicted values
        title: Plot title
        confidence_intervals: (lower, upper) bounds for confidence intervals
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot actual values
    ax.plot(dates, y_true, label='Actual', marker='o', linewidth=2, markersize=4)
    
    # Plot predictions
    ax.plot(dates, y_pred, label='Forecast', marker='s', linewidth=2, markersize=4, alpha=0.8)
    
    # Plot confidence intervals if provided
    if confidence_intervals is not None:
        lower, upper = confidence_intervals
        ax.fill_between(dates, lower, upper, alpha=0.2, label='95% CI')
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: pd.DatetimeIndex = None,
    figsize: tuple = (14, 8)
) -> plt.Figure:
    """
    Plot residual diagnostics.
    
    Creates 4 subplots:
        1. Residuals over time
        2. Residuals vs predictions (homoscedasticity check)
        3. Histogram of residuals
        4. Q-Q plot
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Residuals over time
    if dates is not None:
        axes[0, 0].plot(dates, residuals, marker='o', linestyle='-', alpha=0.6)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0, 0].set_xlabel('Date')
    else:
        axes[0, 0].plot(residuals, marker='o', linestyle='-', alpha=0.6)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0, 0].set_xlabel('Index')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals Over Time')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Residuals vs Predictions
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Predicted Values')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residuals vs Predictions')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Histogram
    axes[1, 0].hist(residuals, bins=20, edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Residuals')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title(f'Histogram (μ={residuals.mean():.3f}, σ={residuals.std():.3f})')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Q-Q Plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot (Normality Check)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def compare_models(
    models_results: Dict[str, Dict[str, float]],
    metrics: List[str] = ['mae', 'rmse', 'r2'],
    figsize: tuple = (12, 6)
) -> plt.Figure:
    """
    Create comparison plot for multiple models.
    
    Parameters:
        models_results: Dict with model names as keys and metric dicts as values
                       e.g., {'SARIMA': {'mae': 0.5, 'rmse': 0.7, ...}, ...}
        metrics: List of metrics to compare
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    df = pd.DataFrame(models_results).T[metrics]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(df.index))
    width = 0.8 / len(metrics)
    
    for i, metric in enumerate(metrics):
        offset = width * i - (width * len(metrics) / 2)
        ax.bar(x + offset, df[metric], width, label=metric.upper())
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Error Value', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df.index)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


def create_metrics_table(
    models_results: Dict[str, Dict[str, float]],
    output_path: str = None
) -> pd.DataFrame:
    """
    Create formatted metrics comparison table.
    
    Parameters:
        models_results: Dict with model names as keys and metric dicts as values
        output_path: Optional path to save CSV
        
    Returns:
        DataFrame with formatted metrics
    """
    df = pd.DataFrame(models_results).T
    
    # Round to 4 decimals
    df = df.round(4)
    
    # Highlight best values (min for errors, max for R²)
    styled_df = df.copy()
    
    if output_path:
        styled_df.to_csv(output_path)
        logger.info(f"Metrics table saved to {output_path}")
    
    return styled_df


def print_metrics_summary(metrics: Dict[str, float], model_name: str = 'Model'):
    """Pretty print metrics summary"""
    logger.info(f"\n{'='*60}")
    logger.info(f"{model_name:^60}")
    logger.info(f"{'='*60}")
    
    for metric_name, value in metrics.items():
        if 'r2' in metric_name.lower():
            # R² should be high
            logger.info(f"{metric_name.upper():20s}: {value:8.4f}")
        else:
            # Errors should be low
            logger.info(f"{metric_name.upper():20s}: {value:8.4f}")
    
    logger.info(f"{'='*60}\n")
