"""
Preprocessing Utilities
Functions for data cleaning and monthly aggregation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def load_raw_weather_data(filepath: str) -> pd.DataFrame:
    """
    Load raw daily weather data from CSV.
    
    Parameters:
        filepath: Path to raw weather CSV
        
    Returns:
        DataFrame with datetime index
    """
    df = pd.read_csv(filepath, parse_dates=['date'])
    df = df.set_index('date').sort_index()
    
    logger.info(f"Loaded {len(df)} days of data ({df.index.min()} to {df.index.max()})")
    return df


def handle_missing_values(
    df: pd.DataFrame,
    max_ffill_days: int = 5,
    precip_fill_value: float = 0.0
) -> pd.DataFrame:
    """
    Handle missing values in weather data.
    
    Strategy:
        - Temperature, ET0, radiation: forward fill up to max_ffill_days
        - Precipitation: fill with 0 (assumption: no rain = 0mm)
        - Wind: forward fill
    
    Parameters:
        df: Raw daily DataFrame
        max_ffill_days: Maximum days to forward fill
        precip_fill_value: Value to fill missing precipitation
        
    Returns:
        DataFrame with missing values handled
    """
    df = df.copy()
    
    # Forward fill temperature, ET0, radiation
    temp_cols = [col for col in df.columns if 'temperature' in col.lower()]
    et0_cols = [col for col in df.columns if 'et0' in col.lower()]
    rad_cols = [col for col in df.columns if 'radiation' in col.lower() or 'shortwave' in col.lower()]
    
    for col in temp_cols + et0_cols + rad_cols:
        if col in df.columns:
            df[col] = df[col].ffill(limit=max_ffill_days)
    
    # Fill precipitation with 0
    precip_cols = [col for col in df.columns if 'precipitation' in col.lower()]
    for col in precip_cols:
        if col in df.columns:
            df[col] = df[col].fillna(precip_fill_value)
    
    # Forward fill wind
    wind_cols = [col for col in df.columns if 'wind' in col.lower()]
    for col in wind_cols:
        if col in df.columns:
            df[col] = df[col].ffill(limit=max_ffill_days)
    
    missing_after = df.isnull().sum()
    if missing_after.sum() > 0:
        logger.warning(f"Remaining missing values:\n{missing_after[missing_after > 0]}")
    
    return df


def aggregate_to_monthly(
    df: pd.DataFrame,
    aggregation_rules: Dict[str, str] = None
) -> pd.DataFrame:
    """
    Aggregate daily data to monthly.
    
    CRITICAL: ET₀ must be summed (mm/month), not averaged!
    
    Parameters:
        df: Daily DataFrame with datetime index
        aggregation_rules: Dict mapping column patterns to agg functions
                          e.g., {'et0': 'sum', 'temperature': 'mean'}
        
    Returns:
        Monthly aggregated DataFrame
    """
    if aggregation_rules is None:
        aggregation_rules = {
            'et0': 'sum',
            'temperature': 'mean',
            'precipitation': 'sum',
            'radiation': 'sum',
            'shortwave': 'sum',
            'wind': 'max'
        }
    
    # Build aggregation dict for each column
    agg_dict = {}
    for col in df.columns:
        col_lower = col.lower()
        
        # Find matching rule
        for pattern, func in aggregation_rules.items():
            if pattern in col_lower:
                agg_dict[col] = func
                break
        else:
            # Default: mean for unknown columns
            agg_dict[col] = 'mean'
            logger.warning(f"No rule for '{col}', using mean")
    
    # Resample to monthly
    monthly = df.resample('MS').agg(agg_dict)
    
    logger.info(f"Aggregated to {len(monthly)} months ({monthly.index.min()} to {monthly.index.max()})")
    
    return monthly


def calculate_water_balance(
    precipitation: pd.Series,
    et0: pd.Series
) -> pd.Series:
    """
    Calculate monthly water balance.
    
    Water Balance = Precipitation - ET₀ (mm/month)
    
    Positive: Water surplus
    Negative: Water deficit
    """
    return precipitation - et0


def calculate_aridity_index(
    precipitation: pd.Series,
    et0: pd.Series,
    epsilon: float = 0.1
) -> pd.Series:
    """
    Calculate aridity index.
    
    Aridity Index = P / ET₀
    
    Values:
        < 0.05: Hyper-arid
        0.05-0.20: Arid
        0.20-0.50: Semi-arid
        0.50-0.65: Dry sub-humid
        > 0.65: Humid
    """
    return precipitation / (et0 + epsilon)


def create_anomalies(
    df: pd.DataFrame,
    columns: list,
    baseline_start: str = None,
    baseline_end: str = None
) -> pd.DataFrame:
    """
    Calculate anomalies relative to baseline period.
    
    Anomaly = Value - Baseline Mean
    
    Parameters:
        df: DataFrame with datetime index
        columns: Columns to calculate anomalies for
        baseline_start: Start of baseline period (None = use all)
        baseline_end: End of baseline period (None = use all)
        
    Returns:
        DataFrame with anomaly columns added
    """
    df = df.copy()
    
    for col in columns:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found, skipping")
            continue
        
        # Calculate baseline statistics
        if baseline_start and baseline_end:
            baseline = df.loc[baseline_start:baseline_end, col]
        else:
            baseline = df[col]
        
        baseline_mean = baseline.mean()
        baseline_std = baseline.std()
        
        # Calculate anomalies
        df[f'{col}_anomaly'] = df[col] - baseline_mean
        df[f'{col}_anomaly_std'] = (df[col] - baseline_mean) / baseline_std
        
        logger.info(f"Created anomalies for '{col}' (baseline: {baseline_mean:.2f} ± {baseline_std:.2f})")
    
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based features for modeling.
    
    Features:
        - month: 1-12
        - quarter: 1-4
        - year: YYYY
        - season: 0-3 (winter, spring, summer, fall)
    """
    df = df.copy()
    
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['year'] = df.index.year
    
    # Season mapping (Northern Hemisphere)
    season_map = {12: 0, 1: 0, 2: 0,  # Winter
                  3: 1, 4: 1, 5: 1,     # Spring
                  6: 2, 7: 2, 8: 2,     # Summer
                  9: 3, 10: 3, 11: 3}   # Fall
    
    df['season'] = df['month'].map(season_map)
    
    return df


def validate_monthly_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate monthly aggregated data.
    
    Checks:
        - Correct frequency (monthly start)
        - No missing dates
        - ET₀ values in reasonable range (sum should be ~40-220 mm/month)
        - Temperature values reasonable
        
    Returns:
        Validation report dict
    """
    report = {
        'valid': True,
        'warnings': [],
        'errors': []
    }
    
    # Check frequency
    inferred_freq = pd.infer_freq(df.index)
    if inferred_freq != 'MS':
        report['warnings'].append(f"Frequency is '{inferred_freq}', expected 'MS'")
    
    # Check for gaps
    expected_dates = pd.date_range(df.index.min(), df.index.max(), freq='MS')
    missing_dates = expected_dates.difference(df.index)
    if len(missing_dates) > 0:
        report['errors'].append(f"Missing {len(missing_dates)} months: {missing_dates.tolist()}")
        report['valid'] = False
    
    # Check ET₀ values (if exists)
    et0_cols = [col for col in df.columns if 'et0' in col.lower()]
    for col in et0_cols:
        et0_values = df[col]
        if et0_values.min() < 10:
            report['warnings'].append(
                f"{col} has suspiciously low values (min={et0_values.min():.1f}). "
                "Expected monthly sum ~40-220 mm. Did you use mean instead of sum?"
            )
        if et0_values.max() > 300:
            report['warnings'].append(f"{col} has very high values (max={et0_values.max():.1f})")
    
    # Check temperature
    temp_cols = [col for col in df.columns if 'temperature' in col.lower() and 'anomaly' not in col.lower()]
    for col in temp_cols:
        temp_values = df[col]
        if temp_values.min() < -20 or temp_values.max() > 50:
            report['warnings'].append(
                f"{col} out of typical range (min={temp_values.min():.1f}, max={temp_values.max():.1f})"
            )
    
    return report
