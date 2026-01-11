"""
Utility Functions
Common helper functions used across the project.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_csv_with_dates(filepath: Path, date_col: str = 'date') -> pd.DataFrame:
    """Load CSV with automatic date parsing"""
    df = pd.read_csv(filepath, parse_dates=[date_col])
    logger.info(f"Loaded {len(df)} rows from {filepath}")
    return df


def save_forecast(
    forecast_df: pd.DataFrame,
    filename: str,
    output_dir: Path = Path('../Predictions')
) -> Path:
    """Save forecast DataFrame to CSV"""
    ensure_dir(output_dir)
    output_path = output_dir / filename
    forecast_df.to_csv(output_path, index=False)
    logger.info(f"Saved forecast to {output_path}")
    return output_path


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculate common forecast evaluation metrics"""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    metrics = {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred),
    }
    
    # MAPE (avoid division by zero)
    mask = y_true != 0
    if mask.any():
        metrics['mape'] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        metrics['mape'] = np.nan
    
    return metrics


def print_metrics(metrics: dict, title: str = "Evaluation Metrics"):
    """Pretty print metrics"""
    logger.info(f"\n{'='*50}")
    logger.info(f"{title:^50}")
    logger.info(f"{'='*50}")
    for name, value in metrics.items():
        logger.info(f"{name.upper():10s}: {value:.4f}")
    logger.info(f"{'='*50}\n")


def split_timeseries(
    df: pd.DataFrame,
    test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split time series data chronologically"""
    split_idx = int(len(df) * (1 - test_size))
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    
    logger.info(f"Train: {len(train)} samples ({train.index.min()} to {train.index.max()})")
    logger.info(f"Test:  {len(test)} samples ({test.index.min()} to {test.index.max()})")
    
    return train, test


def compute_forecast_horizon(
    last_date: pd.Timestamp,
    target_date: pd.Timestamp
) -> int:
    """Calculate number of months between two dates"""
    n_months = (target_date.year - last_date.year) * 12 + (target_date.month - last_date.month)
    
    if n_months <= 0:
        raise ValueError(f"Target date {target_date} must be after last date {last_date}")
    
    return n_months


def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility"""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    
    logger.info(f"Random seeds set to {seed}")


def get_data_summary(df: pd.DataFrame) -> dict:
    """Get comprehensive data summary statistics"""
    summary = {
        'n_rows': len(df),
        'n_cols': len(df.columns),
        'date_range': (df.index.min(), df.index.max()) if isinstance(df.index, pd.DatetimeIndex) else None,
        'missing_values': df.isnull().sum().to_dict(),
        'dtypes': df.dtypes.to_dict(),
    }
    
    return summary


def format_date_range(start: pd.Timestamp, end: pd.Timestamp) -> str:
    """Format date range for display"""
    return f"{start.strftime('%Y-%m')} to {end.strftime('%Y-%m')}"
