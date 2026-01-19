"""
Stationarity Tests for Time Series Data
Implements ADF and KPSS tests as per Phase 1.4 of guide.md
"""
import logging
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss

logger = logging.getLogger(__name__)


def test_stationarity(
    series: pd.Series,
    alpha: float = 0.05,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Perform ADF and KPSS stationarity tests on a time series.
    
    The Augmented Dickey-Fuller (ADF) test checks for unit root (non-stationarity).
    The KPSS test checks for trend stationarity.
    
    Statistical Formulas:
    --------------------
    ADF Test Statistic:
        ΔY_t = α + βt + γY_{t-1} + δ_1ΔY_{t-1} + ... + δ_pΔY_{t-p} + ε_t
        H0: γ = 0 (unit root exists, series is non-stationary)
        H1: γ < 0 (series is stationary)
    
    KPSS Test Statistic:
        LM = (1/T²) Σ(S_t²) / s²(l)
        where S_t = Σ(e_i) are partial sums of residuals
        H0: Series is trend stationary
        H1: Series has a unit root
    
    Parameters
    ----------
    series : pd.Series
        Time series data to test
    alpha : float, default=0.05
        Significance level for hypothesis tests
    verbose : bool, default=True
        If True, log test results
    
    Returns
    -------
    dict
        Dictionary containing:
        - adf_statistic: ADF test statistic
        - adf_pvalue: ADF p-value
        - adf_is_stationary: True if series is stationary (reject H0)
        - kpss_statistic: KPSS test statistic
        - kpss_pvalue: KPSS p-value
        - kpss_is_stationary: True if series is stationary (fail to reject H0)
        - is_stationary: True if both tests agree series is stationary
        - recommendation: String recommendation for differencing
    
    References
    ----------
    - Dickey, D. A., & Fuller, W. A. (1979). Distribution of the estimators for 
      autoregressive time series with a unit root. JASA, 74(366a), 427-431.
    - Kwiatkowski, D., et al. (1992). Testing the null hypothesis of stationarity 
      against the alternative of a unit root. Journal of econometrics, 54(1-3), 159-178.
    """
    # Drop NaN values
    clean_series = series.dropna()
    
    if len(clean_series) < 10:
        logger.warning(f"Series too short ({len(clean_series)} observations) for reliable testing")
        return {
            'error': 'Insufficient data',
            'is_stationary': False
        }
    
    # ADF Test
    try:
        adf_result = adfuller(clean_series, autolag='AIC')
        adf_statistic = adf_result[0]
        adf_pvalue = adf_result[1]
        adf_is_stationary = adf_pvalue < alpha
        
        if verbose:
            logger.info(f"ADF Test: statistic={adf_statistic:.4f}, p-value={adf_pvalue:.4f}")
    except Exception as e:
        logger.error(f"ADF test failed: {e}")
        adf_statistic = adf_pvalue = None
        adf_is_stationary = False
    
    # KPSS Test
    try:
        kpss_result = kpss(clean_series, regression='ct', nlags='auto')
        kpss_statistic = kpss_result[0]
        kpss_pvalue = kpss_result[1]
        kpss_is_stationary = kpss_pvalue > alpha  # Note: reversed logic
        
        if verbose:
            logger.info(f"KPSS Test: statistic={kpss_statistic:.4f}, p-value={kpss_pvalue:.4f}")
    except Exception as e:
        logger.error(f"KPSS test failed: {e}")
        kpss_statistic = kpss_pvalue = None
        kpss_is_stationary = False
    
    # Combined decision
    is_stationary = adf_is_stationary and kpss_is_stationary
    
    # Recommendation
    if is_stationary:
        recommendation = "Series is stationary. No differencing needed (d=0)."
    elif adf_is_stationary and not kpss_is_stationary:
        recommendation = "Series is difference-stationary. Apply first differencing (d=1)."
    elif not adf_is_stationary and kpss_is_stationary:
        recommendation = "Series is trend-stationary. Consider detrending instead of differencing."
    else:
        recommendation = "Series is non-stationary. Apply first differencing (d=1) and retest."
    
    return {
        'adf_statistic': adf_statistic,
        'adf_pvalue': adf_pvalue,
        'adf_is_stationary': adf_is_stationary,
        'kpss_statistic': kpss_statistic,
        'kpss_pvalue': kpss_pvalue,
        'kpss_is_stationary': kpss_is_stationary,
        'is_stationary': is_stationary,
        'recommendation': recommendation
    }


def determine_differencing_order(
    series: pd.Series,
    max_d: int = 2,
    alpha: float = 0.05
) -> Tuple[int, Dict[str, Any]]:
    """
    Determine the optimal differencing order (d) for ARIMA modeling.
    
    Iteratively applies differencing until series becomes stationary or max_d is reached.
    
    Parameters
    ----------
    series : pd.Series
        Time series data
    max_d : int, default=2
        Maximum differencing order to test
    alpha : float, default=0.05
        Significance level for stationarity tests
    
    Returns
    -------
    tuple
        (optimal_d, test_results)
        - optimal_d: Recommended differencing order (0, 1, or 2)
        - test_results: Dictionary of test results for each d
    
    Example
    -------
    >>> d, results = determine_differencing_order(temperature_series)
    >>> print(f"Recommended d={d}")
    >>> print(results[d]['recommendation'])
    """
    test_results = {}
    current_series = series.copy()
    
    for d in range(max_d + 1):
        logger.info(f"Testing differencing order d={d}")
        
        # Test current series
        result = test_stationarity(current_series, alpha=alpha, verbose=False)
        test_results[d] = result
        
        # If stationary, return current d
        if result.get('is_stationary', False):
            logger.info(f"✅ Series is stationary at d={d}")
            return d, test_results
        
        # Apply differencing for next iteration
        if d < max_d:
            current_series = current_series.diff().dropna()
    
    # If still not stationary, return max_d
    logger.warning(f"Series not stationary even at d={max_d}. Using d={max_d}")
    return max_d, test_results


def seasonal_decompose_test(
    series: pd.Series,
    period: int = 12,
    model: str = 'additive'
) -> Dict[str, pd.Series]:
    """
    Perform seasonal decomposition and test components for stationarity.
    
    Decomposition Model:
    -------------------
    Additive: Y_t = T_t + S_t + R_t
    Multiplicative: Y_t = T_t × S_t × R_t
    
    where:
    - Y_t: Observed value
    - T_t: Trend component
    - S_t: Seasonal component
    - R_t: Residual component
    
    Parameters
    ----------
    series : pd.Series
        Time series with DatetimeIndex
    period : int, default=12
        Seasonal period (12 for monthly data)
    model : str, default='additive'
        'additive' or 'multiplicative'
    
    Returns
    -------
    dict
        Dictionary with 'trend', 'seasonal', 'residual' components
        and stationarity test results for each
    """
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    # Perform decomposition
    decomposition = seasonal_decompose(
        series.dropna(),
        model=model,
        period=period,
        extrapolate_trend='freq'
    )
    
    # Test each component
    results = {
        'trend': decomposition.trend,
        'seasonal': decomposition.seasonal,
        'residual': decomposition.resid,
        'trend_test': test_stationarity(decomposition.trend.dropna(), verbose=False),
        'seasonal_test': test_stationarity(decomposition.seasonal.dropna(), verbose=False),
        'residual_test': test_stationarity(decomposition.resid.dropna(), verbose=False)
    }
    
    return results
