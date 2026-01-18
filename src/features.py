"""
Feature Engineering Utilities
Functions for creating drought indices and other derived features.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


def calculate_spi(
    precipitation: pd.Series,
    window: int = 12,
    distribution: str = 'gamma'
) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate Standardized Precipitation Index (SPI).
    
    SPI standardizes precipitation over a specified window, allowing
    comparison of wetness/dryness across different locations and times.
    
    Interpretation:
        SPI > 2.0:  Extremely wet
        SPI > 1.5:  Very wet
        SPI > 1.0:  Moderately wet
        -1.0 < SPI < 1.0: Near normal
        SPI < -1.0: Moderately dry
        SPI < -1.5: Severely dry
        SPI < -2.0: Extremely dry
    
    Parameters:
        precipitation: Monthly precipitation series
        window: Rolling window in months (3, 6, 12, 24 common)
        distribution: 'gamma' or 'normal' for fitting
        
    Returns:
        (SPI values, rolling sum precipitation)
    """
    # Calculate rolling sum
    precip_rolling = precipitation.rolling(window=window, min_periods=window).sum()
    
    # Remove NaN values for fitting
    precip_valid = precip_rolling.dropna()
    
    if len(precip_valid) < 30:
        logger.warning(f"Insufficient data for SPI calculation (n={len(precip_valid)})")
        return pd.Series(np.nan, index=precipitation.index), precip_rolling
    
    # Fit distribution and calculate CDF
    if distribution == 'gamma':
        # Gamma distribution (common for precipitation)
        # Add small constant to avoid zeros
        precip_fit = precip_valid + 0.001
        
        # Fit gamma parameters
        shape, loc, scale = stats.gamma.fit(precip_fit, floc=0)
        
        # Calculate cumulative probabilities
        cdf_values = stats.gamma.cdf(precip_rolling + 0.001, shape, loc, scale)
    else:
        # Normal distribution
        mean = precip_valid.mean()
        std = precip_valid.std()
        cdf_values = stats.norm.cdf(precip_rolling, mean, std)
    
    # Convert to standard normal (Z-scores)
    # Clip to avoid extreme values at boundaries
    cdf_values = np.clip(cdf_values, 0.0001, 0.9999)
    spi = pd.Series(stats.norm.ppf(cdf_values), index=precip_rolling.index)
    
    logger.info(f"Calculated SPI-{window} (range: {spi.min():.2f} to {spi.max():.2f})")
    
    return spi, precip_rolling


def calculate_spei(
    precipitation: pd.Series,
    et0: pd.Series,
    window: int = 12,
    distribution: str = 'log_logistic'
) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate Standardized Precipitation-Evapotranspiration Index (SPEI).
    
    SPEI is similar to SPI but includes evapotranspiration, providing
    a more complete picture of water balance and drought.
    
    Parameters:
        precipitation: Monthly precipitation series
        et0: Monthly reference evapotranspiration series
        window: Rolling window in months
        distribution: 'log_logistic' (recommended) or 'normal'
        
    Returns:
        (SPEI values, rolling water balance)
    """
    # Calculate water balance
    water_balance = precipitation - et0
    
    # Calculate rolling sum
    wb_rolling = water_balance.rolling(window=window, min_periods=window).sum()
    
    # Remove NaN values
    wb_valid = wb_rolling.dropna()
    
    if len(wb_valid) < 30:
        logger.warning(f"Insufficient data for SPEI calculation (n={len(wb_valid)})")
        return pd.Series(np.nan, index=precipitation.index), wb_rolling
    
    if distribution == 'log_logistic':
        # Three-parameter log-logistic distribution (Vicente-Serrano et al., 2010)
        # This is the recommended distribution for SPEI
        
        # Fit log-logistic using method of L-moments (simplified)
        # For simplicity, we'll use normal approximation here
        # In production, use specialized SPEI library
        mean = wb_valid.mean()
        std = wb_valid.std()
        cdf_values = stats.norm.cdf(wb_rolling, mean, std)
    else:
        # Normal distribution
        mean = wb_valid.mean()
        std = wb_valid.std()
        cdf_values = stats.norm.cdf(wb_rolling, mean, std)
    
    # Convert to standard normal
    cdf_values = np.clip(cdf_values, 0.0001, 0.9999)
    spei = pd.Series(stats.norm.ppf(cdf_values), index=wb_rolling.index)
    
    logger.info(f"Calculated SPEI-{window} (range: {spei.min():.2f} to {spei.max():.2f})")
    
    return spei, wb_rolling


def identify_drought_events(
    index: pd.Series,
    threshold: float = -1.0,
    min_duration: int = 3
) -> pd.DataFrame:
    """
    Identify drought events from SPI/SPEI index.
    
    A drought event is a continuous period where index < threshold.
    
    Parameters:
        index: SPI or SPEI series
        threshold: Drought threshold (default: -1.0 = moderate)
        min_duration: Minimum duration in months
        
    Returns:
        DataFrame with columns: start_date, end_date, duration, severity
    """
    drought_mask = index < threshold
    
    # Find drought periods
    events = []
    in_drought = False
    start_date = None
    
    for date, is_drought in drought_mask.items():
        if is_drought and not in_drought:
            # Drought starts
            start_date = date
            in_drought = True
        elif not is_drought and in_drought:
            # Drought ends
            end_date = date
            duration = len(index.loc[start_date:end_date]) - 1
            
            if duration >= min_duration:
                severity = index.loc[start_date:end_date].sum()
                events.append({
                    'start_date': start_date,
                    'end_date': end_date,
                    'duration': duration,
                    'severity': abs(severity),
                    'min_index': index.loc[start_date:end_date].min()
                })
            
            in_drought = False
    
    # Check if drought extends to end of series
    if in_drought and start_date is not None:
        end_date = index.index[-1]
        duration = len(index.loc[start_date:end_date])
        if duration >= min_duration:
            severity = index.loc[start_date:end_date].sum()
            events.append({
                'start_date': start_date,
                'end_date': end_date,
                'duration': duration,
                'severity': abs(severity),
                'min_index': index.loc[start_date:end_date].min()
            })
    
    events_df = pd.DataFrame(events)
    
    if len(events_df) > 0:
        logger.info(f"Found {len(events_df)} drought events (threshold={threshold})")
    else:
        logger.info(f"No drought events found (threshold={threshold})")
    
    return events_df


def create_lagged_features(
    df: pd.DataFrame,
    columns: list,
    lags: list = [1, 3, 6, 12]
) -> pd.DataFrame:
    """
    Create lagged features for time series modeling.
    
    Parameters:
        df: DataFrame with time series data
        columns: Columns to create lags for
        lags: List of lag periods (in months)
        
    Returns:
        DataFrame with lagged columns added
    """
    df = df.copy()
    
    for col in columns:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found, skipping")
            continue
        
        for lag in lags:
            lag_col_name = f'{col}_lag{lag}'
            df[lag_col_name] = df[col].shift(lag)
            
    logger.info(f"Created {len(columns) * len(lags)} lagged features")
    
    return df


def create_rolling_features(
    df: pd.DataFrame,
    columns: list,
    windows: list = [3, 6, 12],
    functions: list = ['mean', 'std']
) -> pd.DataFrame:
    """
    Create rolling window features.
    
    Parameters:
        df: DataFrame with time series data
        columns: Columns to create rolling features for
        windows: Window sizes in months
        functions: Aggregation functions ('mean', 'std', 'min', 'max')
        
    Returns:
        DataFrame with rolling features added
    """
    df = df.copy()
    
    for col in columns:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found, skipping")
            continue
        
        for window in windows:
            for func in functions:
                feature_name = f'{col}_roll{window}_{func}'
                
                if func == 'mean':
                    df[feature_name] = df[col].rolling(window=window, min_periods=1).mean()
                elif func == 'std':
                    df[feature_name] = df[col].rolling(window=window, min_periods=1).std()
                elif func == 'min':
                    df[feature_name] = df[col].rolling(window=window, min_periods=1).min()
                elif func == 'max':
                    df[feature_name] = df[col].rolling(window=window, min_periods=1).max()
    
    logger.info(f"Created {len(columns) * len(windows) * len(functions)} rolling features")
    
    return df
