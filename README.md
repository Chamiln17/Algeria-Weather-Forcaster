# 🇩🇿 Climate Forecasting with Machine Learning & RAG

A comprehensive climate change analysis system for Algeria using statistical models, machine learning forecasting, and AI-powered report generation.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Module Reference](#module-reference)
- [Usage Examples](#usage-examples)
- [API Reference](#api-reference)

---

## Overview

This project analyzes climate data for Algeria, implementing:
- **Statistical Analysis**: Mann-Kendall trends, Sen's slope, stationarity tests
- **Forecasting Models**: SARIMA, Linear Regression, LSTM
- **Drought Indices**: SPI (Standardized Precipitation Index), SPEI
- **AI Reporting**: RAG system with ChromaDB + Groq for natural language queries

---

## Features

| Feature              | Description                                                      |
| -------------------- | ---------------------------------------------------------------- |
| 📊 Data Preprocessing | Missing value handling, monthly aggregation, feature engineering |
| 📈 Trend Analysis     | Mann-Kendall test, Sen's slope estimation                        |
| 🔬 Stationarity Tests | ADF and KPSS tests with differencing recommendations             |
| 🔮 Forecasting        | SARIMA (auto-tuned), Linear baseline, LSTM                       |
| 🌵 Drought Indices    | SPI and SPEI calculation                                         |
| 🤖 AI Reporting       | ChromaDB + e5-small embeddings + Groq Llama 3.1                  |
| 🖥️ Web UI             | Streamlit dashboard with data explorer                           |

---

## Installation

### Prerequisites
- Python 3.11+
- Virtual environment (recommended)

### Setup

```bash
# Clone repository
git clone <repo-url>
cd Project-MLA

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Get Groq API Key (Free)

1. Go to https://console.groq.com/keys
2. Create free account (no credit card required)
3. Copy API key → paste in `.env` file:
   ```
   GROQ_API_KEY=your_key_here
   ```

---

## Quick Start

```bash
# 1. Generate trend analysis
python src/generate_trends.py

# 2. Generate stats database for RAG
python src/generate_stats_db.py

# 3. Run full pipeline test
python test_pipeline.py

# 4. Launch Streamlit app
streamlit run src/app.py
```

Open http://localhost:8501 → Enter Groq API key → Ask questions!

---

## Project Structure

```
Project-MLA/
├── src/                      # Core Python modules
│   ├── preprocessing.py      # Data cleaning & aggregation
│   ├── features.py           # Drought indices & trends
│   ├── stationarity.py       # ADF/KPSS tests
│   ├── forecasting.py        # SARIMA, Linear, LSTM classes
│   ├── rag.py                # ChromaDB + Groq RAG system
│   ├── app.py                # Streamlit UI
│   ├── generate_trends.py    # Trend analysis script
│   ├── generate_stats_db.py  # Stats aggregation script
│   ├── config.py             # Configuration loader
│   ├── evaluation.py         # Forecast metrics
│   └── utils.py              # Helper functions
├── Dataset/                  # Raw data files
├── Preprocessed_dataset/     # Cleaned monthly data
├── Predictions/              # Forecast outputs
├── Results/                  # Analysis outputs
│   ├── trends.json           # Trend analysis results
│   ├── stats_db.json         # Aggregated stats for RAG
│   └── chroma_db/            # Vector database
├── Forecasting_Models/       # Jupyter notebooks
├── config.yaml               # Project configuration
├── requirements.txt          # Python dependencies
└── test_pipeline.py          # Full pipeline test
```

---

## Module Reference

### `src/preprocessing.py` - Data Preprocessing

Cleans and transforms raw weather data into analysis-ready format.

| Function                               | Description                                             |
| -------------------------------------- | ------------------------------------------------------- |
| `load_raw_weather_data(filepath)`      | Load CSV with datetime index                            |
| `handle_missing_values(df)`            | Fill missing: Temperature→forward fill, Precipitation→0 |
| `aggregate_to_monthly(df)`             | Daily→Monthly aggregation (sum precip, mean temp)       |
| `calculate_water_balance(precip, et0)` | Water Balance = P - ET₀                                 |
| `calculate_aridity_index(precip, et0)` | Aridity Index = P / ET₀                                 |
| `create_anomalies(df, columns)`        | Calculate deviations from baseline                      |
| `add_time_features(df)`                | Add month, quarter, season columns                      |

**Key Equation:**
```
Aridity Index = P / ET₀
  < 0.05: Hyper-arid
  0.05-0.20: Arid
  0.20-0.50: Semi-arid
```

---

### `src/features.py` - Feature Engineering

Calculates drought indices and statistical trends.

| Function                                    | Description                                         |
| ------------------------------------------- | --------------------------------------------------- |
| `calculate_spi(precipitation, window)`      | Standardized Precipitation Index                    |
| `calculate_spei(precip, et0, window)`       | Standardized Precipitation-Evapotranspiration Index |
| `identify_drought_events(index, threshold)` | Find drought periods from SPI/SPEI                  |
| `calculate_mk_trend(series)`                | Mann-Kendall trend test                             |
| `calculate_sens_slope(series)`              | Sen's slope estimator                               |
| `create_lagged_features(df, lags)`          | Create lag features for ML                          |
| `create_rolling_features(df, windows)`      | Rolling mean/std features                           |

**Key Equations:**

**SPI Interpretation:**
```
SPI > 2.0:   Extremely wet
SPI > 1.0:   Moderately wet
-1.0 < SPI < 1.0: Normal
SPI < -1.0:  Moderately dry
SPI < -2.0:  Extremely dry
```

**Mann-Kendall Statistic:**
```
S = Σ Σ sgn(x_j - x_i)  for all i < j
```

---

### `src/stationarity.py` - Stationarity Tests

Tests time series for stationarity to determine ARIMA parameters.

| Function                               | Description                   |
| -------------------------------------- | ----------------------------- |
| `test_stationarity(series)`            | Combined ADF + KPSS test      |
| `determine_differencing_order(series)` | Find optimal d for ARIMA      |
| `seasonal_decompose_test(series)`      | Decompose and test components |

**Example:**
```python
from src.stationarity import test_stationarity

result = test_stationarity(temperature_series)
print(result['recommendation'])
# → "Series is stationary. No differencing needed (d=0)."
```

**Tests Performed:**
- **ADF Test**: H₀ = series has unit root (non-stationary)
- **KPSS Test**: H₀ = series is trend stationary

---

### `src/forecasting.py` - Forecasting Models

Three forecasting models with consistent API.

#### `SarimaForecaster`

SARIMA(p,d,q)(P,D,Q)₁₂ model with auto parameter selection.

```python
from src.forecasting import SarimaForecaster

model = SarimaForecaster(seasonal_period=12)
model.fit(temperature_data)
forecast = model.forecast(steps=60)  # 5 years ahead
```

**Attributes after fitting:**
- `model.order` → (p, d, q)
- `model.seasonal_order` → (P, D, Q, 12)

#### `LinearBaseline`

Simple linear extrapolation y(t) = α + β×t

```python
from src.forecasting import LinearBaseline

model = LinearBaseline()
model.fit(temperature_data)
forecast = model.forecast(steps=60)
print(f"Slope: {model.slope:.4f}")  # °C per month
```

#### `LSTMForecaster`

Deep learning forecaster (requires TensorFlow).

Architecture: `Input → LSTM(50) → Dropout(0.2) → LSTM(50) → Dropout(0.2) → Dense(1)`

```python
from src.forecasting import LSTMForecaster

model = LSTMForecaster(lookback=12, epochs=100)
model.fit(temperature_data)
forecast = model.forecast(steps=60, last_sequence=temp[-12:])
```

---

### `src/rag.py` - AI Report Generation

Retrieval-Augmented Generation using ChromaDB + Groq.

**Components:**
- **ChromaDB**: Local vector database (persistent)
- **e5-small**: Fast embedding model (runs locally)
- **Groq Llama 3.1 8B**: Free-tier LLM (14,400 req/day)

**Class: `ClimateRAG`**

```python
from src.rag import init_rag_system

# Initialize
rag = init_rag_system(groq_api_key="your_key")

# Query
answer = rag.query("What are the temperature trends in Algeria?")
print(answer)
```

**Methods:**
| Method                     | Description                    |
| -------------------------- | ------------------------------ |
| `initialize_embeddings()`  | Load e5-small model            |
| `initialize_chroma(reset)` | Create/load ChromaDB           |
| `load_and_embed_data()`    | Embed trends + forecasts       |
| `query(question)`          | Semantic search + LLM response |

---

### `src/app.py` - Streamlit UI

Web dashboard with three tabs:

| Tab             | Description                      |
| --------------- | -------------------------------- |
| 📊 Dashboard     | View result visualizations       |
| 🤖 AI Report     | Ask questions about climate data |
| 📁 Data Explorer | Browse CSV files                 |

**Launch:**
```bash
streamlit run src/app.py
```

---

### `src/generate_trends.py` - Trend Generation

Calculates Mann-Kendall trends for all climate variables.

**Usage:**
```bash
python src/generate_trends.py
```

**Output:** `Results/trends.json`
```json
{
  "temperature_2m_mean": {
    "trend": "increasing",
    "p": 0.0023,
    "h": true,
    "slope": 0.0015
  }
}
```

---

### `src/generate_stats_db.py` - Stats Aggregation

Aggregates trends and forecasts for RAG.

**Usage:**
```bash
python src/generate_stats_db.py
```

**Output:** `Results/stats_db.json`

---

## Usage Examples

### Complete Analysis Pipeline

```python
import pandas as pd
from src.preprocessing import load_raw_weather_data, aggregate_to_monthly
from src.features import calculate_spi, calculate_mk_trend
from src.stationarity import determine_differencing_order
from src.forecasting import SarimaForecaster

# 1. Load and preprocess
df = pd.read_csv('Preprocessed_dataset/algiers_monthly_processed_v2.csv')
temp = df['temperature_2m_mean']

# 2. Calculate drought index
spi, _ = calculate_spi(df['precipitation_sum'], window=12)

# 3. Analyze trends
trend = calculate_mk_trend(temp)
print(f"Trend: {trend['trend']}, p-value: {trend['p']:.4f}")

# 4. Check stationarity
d, _ = determine_differencing_order(temp)
print(f"Use d={d} for ARIMA")

# 5. Forecast
model = SarimaForecaster()
model.fit(temp)
forecast = model.forecast(steps=60)
print(f"Forecast: {forecast['forecast'].head()}")
```

### AI Climate Report

```python
from src.rag import init_rag_system

# Initialize (first time downloads embedding model)
rag = init_rag_system("your_groq_api_key")

# Ask questions
questions = [
    "What are the main temperature trends in Algeria?",
    "Is there evidence of increasing drought?",
    "What do the forecasts predict for 2040?"
]

for q in questions:
    answer = rag.query(q)
    print(f"Q: {q}\nA: {answer}\n")
```

---

## API Reference

### Configuration

**`config.yaml`** - Project settings
```yaml
data:
  precip_column: "precipitation_sum"
  temp_column: "temperature_2m_mean"
  et0_column: "et0_fao_evapotranspiration_sum"
  
et0_aggregation: "sum"  # CRITICAL: ET₀ must be summed, not averaged

forecasting:
  horizon_months: 60
  seasonal_period: 12
```

### Environment Variables

**`.env`** (create this file)
```
GROQ_API_KEY=your_groq_api_key_here
```

---

## Testing

Run the comprehensive pipeline test:

```bash
python test_pipeline.py
```

Tests:
- ✅ Data loading
- ✅ Feature engineering (SPI, SPEI)
- ✅ Trend analysis (Mann-Kendall, Sen's slope)
- ✅ Stationarity tests (ADF, KPSS)
- ✅ Forecasting (SARIMA, Linear)
- ✅ Stats database generation

---

## License

MIT License

---

## Acknowledgments

- Open-Meteo for climate data
- Groq for free LLM API
- ChromaDB for vector storage
- Sentence Transformers for embeddings
