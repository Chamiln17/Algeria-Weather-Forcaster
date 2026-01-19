"""
Forecasting Models for Climate Data
Implements SARIMA, Linear Baseline, and LSTM forecasters as per Phase 3 of guide.md
"""
import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy import stats
import joblib

logger = logging.getLogger(__name__)


class SarimaForecaster:
    """
    SARIMA (Seasonal ARIMA) Forecaster
    
    Implements seasonal autoregressive integrated moving average model for time series forecasting.
    
    Model Equation:
    --------------
    SARIMA(p,d,q)(P,D,Q)_s where:
    - (p,d,q): Non-seasonal parameters (AR order, differencing, MA order)
    - (P,D,Q): Seasonal parameters
    - s: Seasonal period (12 for monthly data)
    
    The model combines:
    1. AR (AutoRegressive): Y_t depends on past values
    2. I (Integrated): Differencing to achieve stationarity
    3. MA (Moving Average): Y_t depends on past forecast errors
    4. Seasonal components for each of the above
    
    References
    ----------
    Box, G. E., Jenkins, G. M., & Reinsel, G. C. (2015). Time series analysis: 
    forecasting and control. John Wiley & Sons.
    """
    
    def __init__(
        self,
        seasonal_period: int = 12,
        auto_select: bool = True,
        order: Optional[Tuple[int, int, int]] = None,
        seasonal_order: Optional[Tuple[int, int, int, int]] = None
    ):
        """
        Initialize SARIMA forecaster.
        
        Parameters
        ----------
        seasonal_period : int, default=12
            Seasonal period (12 for monthly data)
        auto_select : bool, default=True
            If True, use auto_arima to select optimal parameters
        order : tuple, optional
            (p,d,q) order if not using auto_select
        seasonal_order : tuple, optional
            (P,D,Q,s) seasonal order if not using auto_select
        """
        self.seasonal_period = seasonal_period
        self.auto_select = auto_select
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.fitted_model = None
        self.training_data = None
        
    def fit(
        self,
        data: pd.Series,
        exog: Optional[pd.DataFrame] = None,
        **auto_arima_kwargs
    ) -> 'SarimaForecaster':
        """
        Fit SARIMA model to training data.
        
        Parameters
        ----------
        data : pd.Series
            Training time series data
        exog : pd.DataFrame, optional
            Exogenous variables
        **auto_arima_kwargs
            Additional arguments for auto_arima
        
        Returns
        -------
        self
            Fitted forecaster instance
        """
        self.training_data = data.copy()
        
        if self.auto_select:
            logger.info("Running auto_arima to select optimal parameters...")
            
            # Default auto_arima parameters
            default_kwargs = {
                'seasonal': True,
                'm': self.seasonal_period,
                'stepwise': True,
                'suppress_warnings': True,
                'error_action': 'ignore',
                'max_p': 3,
                'max_q': 3,
                'max_P': 2,
                'max_Q': 2,
                'max_d': 2,
                'max_D': 1,
                'trace': False
            }
            default_kwargs.update(auto_arima_kwargs)
            
            self.model = auto_arima(data, **default_kwargs)
            self.order = self.model.order
            self.seasonal_order = self.model.seasonal_order
            
            logger.info(f"Selected SARIMA{self.order}x{self.seasonal_order}")
        else:
            logger.info(f"Fitting SARIMA{self.order}x{self.seasonal_order}")
            self.fitted_model = SARIMAX(
                data,
                exog=exog,
                order=self.order,
                seasonal_order=self.seasonal_order
            ).fit(disp=False)
        
        return self
    
    def forecast(
        self,
        steps: int,
        exog: Optional[pd.DataFrame] = None,
        return_conf_int: bool = True,
        alpha: float = 0.05
    ) -> pd.DataFrame:
        """
        Generate forecasts.
        
        Parameters
        ----------
        steps : int
            Number of steps ahead to forecast
        exog : pd.DataFrame, optional
            Future exogenous variables
        return_conf_int : bool, default=True
            If True, return confidence intervals
        alpha : float, default=0.05
            Significance level for confidence intervals (95% CI)
        
        Returns
        -------
        pd.DataFrame
            Forecast with columns: 'forecast', 'lower_ci', 'upper_ci'
        """
        if self.model is None and self.fitted_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if self.auto_select:
            forecast_result = self.model.predict(
                n_periods=steps,
                return_conf_int=return_conf_int,
                alpha=alpha
            )
            
            if return_conf_int:
                forecast_values, conf_int = forecast_result
                result = pd.DataFrame({
                    'forecast': forecast_values,
                    'lower_ci': conf_int[:, 0],
                    'upper_ci': conf_int[:, 1]
                })
            else:
                result = pd.DataFrame({'forecast': forecast_result})
        else:
            forecast_result = self.fitted_model.get_forecast(
                steps=steps,
                exog=exog
            )
            result = pd.DataFrame({
                'forecast': forecast_result.predicted_mean,
                'lower_ci': forecast_result.conf_int(alpha=alpha).iloc[:, 0],
                'upper_ci': forecast_result.conf_int(alpha=alpha).iloc[:, 1]
            })
        
        return result
    
    def save_predictions(
        self,
        forecast_df: pd.DataFrame,
        output_path: Path,
        variable_name: str
    ):
        """Save forecast to CSV"""
        forecast_df['variable'] = variable_name
        forecast_df.to_csv(output_path, index=False)
        logger.info(f"Saved forecast to {output_path}")


class LinearBaseline:
    """
    Linear Baseline Forecaster
    
    Simple linear extrapolation using ordinary least squares regression.
    
    Model Equation:
    --------------
    y(t) = α + β × t
    
    where:
    - α: Intercept
    - β: Slope (trend)
    - t: Time index
    
    This serves as a baseline for comparison with more complex models.
    """
    
    def __init__(self):
        """Initialize linear forecaster"""
        self.slope = None
        self.intercept = None
        self.training_data = None
        
    def fit(self, data: pd.Series) -> 'LinearBaseline':
        """
        Fit linear trend to data.
        
        Parameters
        ----------
        data : pd.Series
            Training time series data
        
        Returns
        -------
        self
            Fitted forecaster instance
        """
        self.training_data = data.copy()
        
        # Create time index
        x = np.arange(len(data))
        y = data.values
        
        # Fit linear regression
        result = stats.linregress(x, y)
        self.slope = result.slope
        self.intercept = result.intercept
        
        logger.info(f"Linear fit: y = {self.intercept:.4f} + {self.slope:.4f}*t")
        logger.info(f"R² = {result.rvalue**2:.4f}")
        
        return self
    
    def forecast(self, steps: int) -> pd.DataFrame:
        """
        Generate linear forecast.
        
        Parameters
        ----------
        steps : int
            Number of steps ahead to forecast
        
        Returns
        -------
        pd.DataFrame
            Forecast with column 'forecast'
        """
        if self.slope is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Extrapolate
        start_idx = len(self.training_data)
        future_indices = np.arange(start_idx, start_idx + steps)
        forecast_values = self.intercept + self.slope * future_indices
        
        return pd.DataFrame({'forecast': forecast_values})
    
    def save_predictions(
        self,
        forecast_df: pd.DataFrame,
        output_path: Path,
        variable_name: str
    ):
        """Save forecast to CSV"""
        forecast_df['variable'] = variable_name
        forecast_df.to_csv(output_path, index=False)
        logger.info(f"Saved forecast to {output_path}")


class LSTMForecaster:
    """
    LSTM (Long Short-Term Memory) Forecaster
    
    Deep learning model for time series forecasting using LSTM neural networks.
    
    Architecture:
    ------------
    Input(lookback) → LSTM(50) → Dropout(0.2) → LSTM(50) → Dropout(0.2) → Dense(1)
    
    LSTM Cell Equations:
    -------------------
    f_t = σ(W_f · [h_{t-1}, x_t] + b_f)  # Forget gate
    i_t = σ(W_i · [h_{t-1}, x_t] + b_i)  # Input gate
    C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)  # Candidate values
    C_t = f_t * C_{t-1} + i_t * C̃_t  # Cell state
    o_t = σ(W_o · [h_{t-1}, x_t] + b_o)  # Output gate
    h_t = o_t * tanh(C_t)  # Hidden state
    
    Note: Requires TensorFlow/Keras (optional dependency)
    """
    
    def __init__(
        self,
        lookback: int = 12,
        lstm_units: int = 50,
        dropout: float = 0.2,
        epochs: int = 100,
        batch_size: int = 32
    ):
        """
        Initialize LSTM forecaster.
        
        Parameters
        ----------
        lookback : int, default=12
            Number of past time steps to use for prediction
        lstm_units : int, default=50
            Number of LSTM units per layer
        dropout : float, default=0.2
            Dropout rate for regularization
        epochs : int, default=100
            Number of training epochs
        batch_size : int, default=32
            Batch size for training
        """
        self.lookback = lookback
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.scaler = None
        
        # Check if TensorFlow is available
        try:
            import tensorflow as tf
            self.tf_available = True
        except ImportError:
            logger.warning("TensorFlow not installed. LSTM forecaster unavailable.")
            self.tf_available = False
    
    def fit(self, data: pd.Series) -> 'LSTMForecaster':
        """
        Fit LSTM model (requires TensorFlow).
        
        Parameters
        ----------
        data : pd.Series
            Training time series data
        
        Returns
        -------
        self
            Fitted forecaster instance
        """
        if not self.tf_available:
            raise ImportError("TensorFlow required for LSTM. Install with: pip install tensorflow")
        
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from sklearn.preprocessing import MinMaxScaler
        
        # Scale data
        self.scaler = MinMaxScaler()
        scaled_data = self.scaler.fit_transform(data.values.reshape(-1, 1))
        
        # Create sequences
        X, y = [], []
        for i in range(self.lookback, len(scaled_data)):
            X.append(scaled_data[i-self.lookback:i, 0])
            y.append(scaled_data[i, 0])
        X, y = np.array(X), np.array(y)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        # Build model
        self.model = Sequential([
            LSTM(self.lstm_units, return_sequences=True, input_shape=(self.lookback, 1)),
            Dropout(self.dropout),
            LSTM(self.lstm_units),
            Dropout(self.dropout),
            Dense(1)
        ])
        
        self.model.compile(optimizer='adam', loss='mse')
        
        # Train
        logger.info(f"Training LSTM for {self.epochs} epochs...")
        self.model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0
        )
        
        return self
    
    def forecast(self, steps: int, last_sequence: np.ndarray) -> pd.DataFrame:
        """
        Generate LSTM forecast.
        
        Parameters
        ----------
        steps : int
            Number of steps ahead to forecast
        last_sequence : np.ndarray
            Last 'lookback' values from training data
        
        Returns
        -------
        pd.DataFrame
            Forecast with column 'forecast'
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Forecast iteratively
        current_sequence = last_sequence.copy()
        forecasts = []
        
        for _ in range(steps):
            # Predict next value
            scaled_pred = self.model.predict(
                current_sequence.reshape(1, self.lookback, 1),
                verbose=0
            )[0, 0]
            
            # Inverse transform
            pred = self.scaler.inverse_transform([[scaled_pred]])[0, 0]
            forecasts.append(pred)
            
            # Update sequence
            current_sequence = np.append(current_sequence[1:], scaled_pred)
        
        return pd.DataFrame({'forecast': forecasts})
    
    def save_predictions(
        self,
        forecast_df: pd.DataFrame,
        output_path: Path,
        variable_name: str
    ):
        """Save forecast to CSV"""
        forecast_df['variable'] = variable_name
        forecast_df.to_csv(output_path, index=False)
        logger.info(f"Saved forecast to {output_path}")
