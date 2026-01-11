# Climate Forecasting with Machine Learning & Reinforcement Learning

**Master's Project**: Advanced climate time series forecasting for Algiers using SARIMA, Linear Regression, LSTM, and RL-based model selection.

---

## 📋 Project Overview

This project implements a comprehensive climate forecasting pipeline combining:
- **Statistical Methods**: SARIMA, Linear Regression
- **Deep Learning**: LSTM Neural Networks
- **Reinforcement Learning**: Adaptive model selection
- **RAG System**: LLM-based report generation

### Research Questions
1. How will Algiers' temperature evolve through 2040?
2. What are the long-term trends in precipitation and evapotranspiration?
3. Can RL improve forecast accuracy by dynamically selecting models?

---

## 🏗️ Project Structure

```
Project-MLA/
│
├── 📁 Dataset/                 # Raw weather data
├── 📁 Preprocessed_dataset/    # Cleaned monthly data
├── 📁 Forecasting_Models/      # Model notebooks
├── 📁 Models/                  # Saved trained models
├── 📁 Predictions/             # Forecast outputs (CSV)
├── 📁 Results/                 # Plots, figures, reports
├── 📁 src/                     # Utility modules
│
├── verify_phase1_phase2.ipynb  # Reference implementation (Phases 1-2)
├── requirements.txt            # Python dependencies
├── config.yaml                 # Project configuration
└── README.md                   # This file
```

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Complete Pipeline

```bash
# Phase 1-2: Preprocessing & Trend Analysis
jupyter nbconvert --to notebook --execute verify_phase1_phase2.ipynb

# Phase 3: Forecasting Models
jupyter nbconvert --to notebook --execute Forecasting_Models/sarima_final.ipynb
jupyter nbconvert --to notebook --execute Forecasting_Models/linear_final.ipynb
jupyter nbconvert --to notebook --execute Forecasting_Models/lstm_final.ipynb
```

### 3. View Results

Outputs are saved to:
- `Predictions/` — Forecast CSVs (2024-2040)
- `Models/` — Trained model artifacts
- `Results/` — Visualization plots

---

## 📊 Methodology

### Phase 1: Data Preprocessing
- **Input**: Daily weather data (2002-2023) from Open-Meteo
- **Processing**:
  - Missing value imputation (5-day forward fill limit)
  - **Critical fix**: ET₀ aggregated as **sum** (mm/month), not mean
  - Feature engineering: anomalies, water balance, aridity index
- **Output**: `algiers_monthly_processed_v2.csv` (262 months)

### Phase 2: Statistical Analysis
- **Stationarity Tests**: ADF, KPSS
- **Trend Detection**: Mann-Kendall with Hamed-Rao autocorrelation correction
- **Drought Indices**: SPI/SPEI (12-month standardized)
- **Key Finding**: Significant warming trend (+0.034°C/year, p<0.001)

### Phase 3: Forecasting Models

| Model | Description | Horizon |
|-------|-------------|---------|
| **SARIMA** | Seasonal ARIMA (auto-fitted) | 2024-01 → 2040-12 |
| **Linear** | Time trend + seasonal dummies | 2024-01 → 2040-12 |
| **LSTM** | 2-layer recurrent network (64→32 units) | 2024-01 → 2040-12 |

### Phase 4: RL Model Selector *(In Progress)*
- **State**: Recent errors, seasonality, trend
- **Actions**: Choose SARIMA, Linear, or LSTM
- **Reward**: Negative forecast error
- **Algorithm**: Q-learning / DQN

---

## 📈 Key Results

### Temperature Trends (2002-2023)
- **Mean**: 17.8°C
- **Mann-Kendall**: Significant upward trend (p < 0.001)
- **Sen's Slope**: +0.034°C/year (autocorrelation-corrected)

### Forecasts to 2040
- **SARIMA**: Mean 18.5°C (±0.8°C confidence interval)
- **Linear**: Mean 18.7°C (linear extrapolation)
- **LSTM**: Mean 18.4°C (captures non-linearity)

### Drought Analysis
- **Severe droughts** (SPEI < -1.5): 18 months (6.9%)
- **Worst month**: 2023-08 (SPEI = -2.34)

---

## 🛠️ Technologies

- **Python 3.9+**
- **Data**: pandas, numpy
- **Statistics**: statsmodels, pymannkendall, pmdarima
- **ML**: scikit-learn, TensorFlow/Keras
- **RL**: (TBD: Stable-Baselines3 / custom)
- **Viz**: matplotlib, seaborn

---

## 📝 Citation

```bibtex
@mastersthesis{algiers_climate_2026,
  title={Climate Forecasting with Hybrid ML and RL Approaches: A Case Study of Algiers},
  author={[Your Name]},
  year={2026},
  school={[Your University]},
  type={Master's Thesis}
}
```

---

## 📄 License

This project is for academic purposes. Data source: [Open-Meteo](https://open-meteo.com/).

---

## 🤝 Contributors

- **Phase 1-2**: Data preprocessing, trend analysis, drought indices
- **Phase 3**: SARIMA, Linear, LSTM forecasting models
- **Phase 4**: RL integration *(planned)*

---

## 📧 Contact

For questions: [your.email@university.edu]

**Last Updated**: January 2026