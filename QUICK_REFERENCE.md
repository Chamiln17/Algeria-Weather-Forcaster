# 🚀 Quick Reference Card

## Project Files at a Glance

```
Project-MLA/
│
├── 📘 README.md                      ⭐ Start here! Project overview
├── 📘 STRUCTURE.md                   File organization guide
├── 📘 IMPROVEMENTS.md                What we improved & why
├── 📘 requirements.txt               Python dependencies
├── 📘 config.yaml                    Centralized configuration
├── 📘 .gitignore                     Git version control rules
│
├── 📓 example_utilities_usage.ipynb  How to use src/ modules
├── 📓 verify_phase1_phase2.ipynb     Reference implementation
│
├── 📁 src/                           ⭐ Reusable utility modules
│   ├── __init__.py
│   ├── config.py                     Load config.yaml
│   ├── utils.py                      General utilities
│   ├── preprocessing.py              Data cleaning & aggregation
│   ├── features.py                   Drought indices, lags, rolling
│   └── evaluation.py                 Metrics, plots, comparisons
│
├── 📁 Dataset/                       Raw weather data
├── 📁 Preprocessed_dataset/          Monthly processed data
├── 📁 Forecasting_Models/            SARIMA, Linear, LSTM notebooks
├── 📁 Models/                        Saved trained models
├── 📁 Predictions/                   Forecast CSVs (2024-2040)
└── 📁 Results/                       Plots, figures, reports
```

---

## 🎯 Common Tasks

### 1️⃣ Setup Environment
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2️⃣ Use Utilities in Notebooks
```python
import sys
sys.path.append('../src')

from config import get_config, get_path
from utils import load_csv_with_dates, calculate_metrics
from preprocessing import validate_monthly_data
from features import calculate_spi, calculate_spei
from evaluation import plot_forecast_vs_actual

# Load config
config = get_config()
data = load_csv_with_dates(str(get_path('processed_data')))
```

### 3️⃣ Access Configuration
```python
from config import get_config

config = get_config()

# Get paths
config.get('paths.processed_data')      # Data path
config.get('paths.models')              # Model path

# Get parameters
config.get('preprocessing.et0_aggregation')     # 'sum'
config.get('forecasting.lstm.look_back')        # 12
config.get('forecasting.sarima.seasonal')       # true
```

### 4️⃣ Validate Data
```python
from preprocessing import validate_monthly_data

report = validate_monthly_data(df)
print(f"Valid: {report['valid']}")
print(f"Warnings: {report['warnings']}")
```

### 5️⃣ Calculate Drought Indices
```python
from features import calculate_spi, calculate_spei, identify_drought_events

# SPI-12
spi_12, _ = calculate_spi(df['precipitation_sum'], window=12)

# SPEI-12
spei_12, _ = calculate_spei(df['precipitation_sum'], df['et0_fao_sum'], window=12)

# Find drought events
events = identify_drought_events(spei_12, threshold=-1.0, min_duration=3)
```

### 6️⃣ Evaluate Forecasts
```python
from evaluation import calculate_forecast_metrics, plot_forecast_vs_actual, print_metrics_summary

# Calculate metrics
metrics = calculate_forecast_metrics(y_true, y_pred, prefix='test_')
print_metrics_summary(metrics, model_name='SARIMA')

# Plot
fig = plot_forecast_vs_actual(dates, y_true, y_pred, title='Forecast')
fig.savefig('../Results/forecast.png', dpi=300)
```

### 7️⃣ Save Forecasts
```python
from utils import save_forecast
from config import get_path

forecast_df = pd.DataFrame({'date': dates, 'temperature': predictions})
save_forecast(forecast_df, 'sarima_forecast_2040.csv', get_path('predictions'))
```

---

## 📦 Module Functions Quick Reference

### `config.py`
- `get_config()` - Load configuration
- `get_path(key)` - Get path from config
- `get_random_seed(library)` - Get random seed

### `utils.py`
- `load_csv_with_dates(path)` - Load CSV with dates
- `save_forecast(df, filename)` - Save forecast
- `calculate_metrics(y_true, y_pred)` - MAE, RMSE, R², MAPE
- `split_timeseries(df, test_size)` - Train/test split
- `set_random_seeds(seed)` - Set all random seeds

### `preprocessing.py`
- `load_raw_weather_data(path)` - Load raw daily data
- `handle_missing_values(df)` - Impute missing values
- `aggregate_to_monthly(df)` - Daily → Monthly (ET₀ sum!)
- `calculate_water_balance(P, ET0)` - P - ET₀
- `calculate_aridity_index(P, ET0)` - P / ET₀
- `validate_monthly_data(df)` - Data quality checks

### `features.py`
- `calculate_spi(precip, window)` - Standardized Precipitation Index
- `calculate_spei(precip, et0, window)` - SPEI with ET₀
- `identify_drought_events(index)` - Find drought periods
- `create_lagged_features(df, cols, lags)` - Time lags
- `create_rolling_features(df, cols, windows)` - Moving stats

### `evaluation.py`
- `calculate_forecast_metrics(y_true, y_pred)` - All metrics
- `plot_forecast_vs_actual(dates, y_true, y_pred)` - Forecast plot
- `plot_residuals(y_true, y_pred)` - 4-panel diagnostics
- `compare_models(results_dict)` - Multi-model comparison
- `print_metrics_summary(metrics, name)` - Pretty print

---

## 🔧 Configuration Keys (config.yaml)

### Paths
- `paths.raw_data`
- `paths.processed_data`
- `paths.models`
- `paths.predictions`
- `paths.results`

### Preprocessing
- `preprocessing.max_ffill_days` (5)
- `preprocessing.et0_aggregation` ('sum') ⚠️ Critical!
- `preprocessing.temp_aggregation` ('mean')

### Forecasting
- `forecasting.target_date` ('2040-12-01')
- `forecasting.target_variable` ('temperature_2m_mean')
- `forecasting.sarima.seasonal` (true)
- `forecasting.sarima.seasonal_period` (12)
- `forecasting.lstm.look_back` (12)
- `forecasting.lstm.layers` (list of layer configs)

### Random Seeds
- `random_seeds.numpy` (42)
- `random_seeds.tensorflow` (42)
- `random_seeds.sklearn` (42)

---

## 🎨 Visualization Examples

```python
from evaluation import plot_forecast_vs_actual, plot_residuals
import matplotlib.pyplot as plt

# Forecast plot with confidence intervals
fig = plot_forecast_vs_actual(
    dates=test_dates,
    y_true=actual_temps,
    y_pred=forecast_temps,
    title='SARIMA Temperature Forecast',
    confidence_intervals=(lower_bound, upper_bound),
    figsize=(14, 6)
)
plt.savefig('../Results/sarima_forecast.png', dpi=300, bbox_inches='tight')
plt.show()

# Residual diagnostics
fig = plot_residuals(
    y_true=actual_temps,
    y_pred=forecast_temps,
    dates=test_dates,
    figsize=(14, 8)
)
plt.savefig('../Results/sarima_residuals.png', dpi=300, bbox_inches='tight')
plt.show()
```

---

## 📚 Documentation Files

| File | Purpose |
|------|---------|
| **README.md** | Project overview, quick start, methodology |
| **STRUCTURE.md** | Directory organization, naming conventions |
| **IMPROVEMENTS.md** | Detailed changelog, usage examples |
| **QUICK_REFERENCE.md** | This file! Common tasks & functions |
| **example_utilities_usage.ipynb** | Interactive examples |

---

## ⚡ Pro Tips

1. **Always use config.yaml**: Don't hardcode paths or parameters
2. **Use utilities**: Less code duplication = fewer bugs
3. **Set random seeds**: `set_random_seeds(42)` at notebook start
4. **Validate data**: Run `validate_monthly_data()` after preprocessing
5. **Check ET₀**: Monthly sum should be ~40-220 mm, not ~1-6!
6. **Save everything**: Use `save_forecast()` for consistent outputs
7. **Document changes**: Update IMPROVEMENTS.md with your additions

---

## 🆘 Troubleshooting

**Import errors?**
```python
import sys
sys.path.append('../src')  # Add this at notebook start
```

**Config not found?**
- Make sure config.yaml is in project root
- Check notebook location relative to config.yaml
- Use `config_path='../config.yaml'` if needed

**Data validation warnings?**
- Check ET₀ aggregation (should be 'sum', not 'mean')
- Verify date frequency is 'MS' (month start)
- Ensure no missing months in time series

---

**Last Updated**: January 2026  
**For Support**: Check IMPROVEMENTS.md or example_utilities_usage.ipynb
