"""
Algiers Climate Forecasting Project
A comprehensive climate time series analysis and forecasting system.
"""

__version__ = '1.0.0'
__author__ = 'Master\'s Project Team'

# Import main utilities for easy access
from .config import get_config, get_path, get_random_seed
from .utils import (
    ensure_dir,
    load_csv_with_dates,
    save_forecast,
    calculate_metrics,
    set_random_seeds
)

__all__ = [
    'get_config',
    'get_path',
    'get_random_seed',
    'ensure_dir',
    'load_csv_with_dates',
    'save_forecast',
    'calculate_metrics',
    'set_random_seeds',
]
