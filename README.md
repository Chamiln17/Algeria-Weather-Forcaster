# 🇩🇿 Climate Forecasting System for Algeria

> Long-term climate forecasting (2024–2040) using an ensemble of ML models, an RL-based model selector, and a RAG-powered conversational interface — built for the Algiers region.

---

## 🏗️ System Architecture

![System Architecture](Project_Architecture/Architecture.png)

The system is organized as a four-stage pipeline:

```
  Raw Data (Open-Meteo)
        │
        ▼
┌──────────────────────────┐
│   DATA PIPELINE          │  Preprocessing → Monthly Aggregation → Feature Engineering
│   src/preprocessing.py   │  Drought indices (SPI, SPEI), anomalies, rolling stats
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│   FORECASTING MODELS     │  SARIMA │ LSTM │ Ridge │ Prophet
│   Forecasting_Models/    │  Each produces 204-month forecasts (→ 2040)
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│   RL AGENT               │  Q-Learning agent trained on 2019–2023 backcasts
│   RL Agent/              │  Selects BEST model per month (dual-variable reward)
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│   RAG + UI               │  ChromaDB + e5-small-v2 + Groq (Kimi K2)
│   src/rag.py + app.py    │  Streamlit dashboard with streaming answers
└──────────────────────────┘
```

---

## ✨ Features

| Feature                    | Description                                                                                   |
| -------------------------- | --------------------------------------------------------------------------------------------- |
| 📊 **Data Pipeline**        | Missing value handling, monthly aggregation, drought indices (SPI, SPEI), anomaly calculation |
| 📈 **Trend Analysis**       | Mann-Kendall test (with autocorrelation correction), Sen's slope estimation                   |
| 🔬 **Stationarity**         | ADF + KPSS tests with automatic differencing order recommendation                             |
| 🔮 **4 Forecasting Models** | SARIMA (auto-tuned), Bidirectional LSTM, Ridge Regression (polynomial), Prophet               |
| 🤖 **RL Model Selector**    | Q-Learning agent that picks the best model for each calendar month                            |
| 🌵 **Drought Assessment**   | SPI-12 and SPEI-12 with severity classification                                               |
| 💬 **RAG Chat**             | ChromaDB + e5-small-v2 embeddings + Groq Kimi K2 with streaming                               |
| 📊 **Auto Visualizations**  | Forecast plots, model comparisons, uncertainty growth charts                                  |
| 📄 **PDF Reports**          | Automated climate report generation with charts and statistics                                |
| 🖥️ **Streamlit UI**         | Interactive dashboard with data explorer and AI chat                                          |

---

## 🤖 RL Agent — Intelligent Model Selection

![RL Architecture](Project_Architecture/RL_Architecture.png)

The RL Agent is the core innovation — instead of picking one model, it **learns which model performs best for each month** using Q-Learning.

### How It Works

1. **Training (2019–2023):** All 4 models backcast 56 months of known data
2. **Dual-Variable Reward:** Error is computed on **both** Temperature and ET₀, normalized by best-model performance
3. **Q-Table:** 12 states (months) × 4 actions (models) → learns seasonal preferences
4. **Result:** LSTM selected 66.7% of months, SARIMA 33.3% — Ridge and Prophet eliminated

### Model Performance (Backcast 2019–2023)

| Model      | Temp MAE (°C) | ET₀ MAE (mm) | Selected |
| ---------- | ------------- | ------------ | -------- |
| **LSTM**   | 1.084 ⭐       | 15.432 ⭐     | 66.7%    |
| **SARIMA** | 1.106         | 17.385       | 33.3%    |
| Ridge      | 5.174         | 45.061       | 0%       |
| Prophet    | 4.126         | 48.739       | 0%       |

---

## 💬 RAG System

![RAG Architecture](Project_Architecture/RAG.png)

The RAG system lets users ask natural-language questions about climate data.

**Stack:** ChromaDB (vector store) → e5-small-v2 (embeddings) → Groq API with **Kimi K2** (LLM)

**Example:**
```python
from src.rag import init_rag_system

rag = init_rag_system("your_groq_api_key")
answer = rag.query("What will the temperature be in Algeria in 2035?")
```

---

## 📁 Project Structure

```
Project-MLA/
├── src/                          # Core Python modules
│   ├── preprocessing.py          # Data cleaning & monthly aggregation
│   ├── features.py               # Drought indices (SPI, SPEI) & trends
│   ├── stationarity.py           # ADF/KPSS stationarity tests
│   ├── forecasting.py            # SARIMA, Linear, LSTM forecasters
│   ├── evaluation.py             # Forecast metrics (MAE, RMSE, R², MAPE)
│   ├── rag.py                    # ChromaDB + Groq RAG system
│   ├── visualizer.py             # Auto-generated forecast charts
│   ├── report_generator.py       # PDF climate report builder
│   ├── app.py                    # Streamlit UI
│   ├── config.py                 # Configuration loader
│   ├── utils.py                  # Helper functions
│   ├── generate_trends.py        # Mann-Kendall trend analysis
│   └── generate_stats_db.py      # Stats aggregation for RAG
│
├── RL Agent/                     # Reinforcement Learning model selector
│   ├── pretrain_agent.py         # Q-Learning training script
│   ├── rl_forecast_unified.py    # Unified forecast generation
│   ├── agent.py                  # RLAgent class
│   ├── simulate_env.py           # Environment simulation
│   ├── add_uncertainty.py        # Uncertainty quantification
│   ├── analyze_results.py        # Result analysis & plots
│   └── pretrained_q_table.pkl    # Trained Q-table
│
├── Forecasting_Models/           # Jupyter notebooks (training & analysis)
│   ├── sarima_final.ipynb
│   ├── lstm_final.ipynb
│   ├── linear_final.ipynb
│   ├── unified_forecast.ipynb
│   └── backcast_generator.ipynb
│
├── Dataset/                      # Raw data (Open-Meteo API)
├── Preprocessed_dataset/         # Cleaned monthly data
├── Predictions/                  # Model forecast CSVs (→ 2040)
├── Models/                       # Serialized models (.pkl, .h5)
├── Results/                      # Analysis outputs & ChromaDB
│   ├── stats_db.json             # Aggregated stats for RAG
│   ├── chroma_db/                # Vector database
│   └── *.png                     # Generated plots
│
├── Project_Architecture/         # Architecture diagrams
│   ├── Architecture.png
│   ├── RL_Architecture.png
│   └── RAG.png
│
├── config.yaml                   # Project configuration
├── requirements.txt              # Python dependencies
├── test_pipeline.py              # Full pipeline test
└── EXPLANATION.md                # Detailed technical documentation
```

---

## 🚀 Installation

### Prerequisites
- Python 3.11+
- Groq API key (free at [console.groq.com/keys](https://console.groq.com/keys))

### Setup

```bash
# Clone repository
git clone <repo-url>
cd Project-MLA

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file:
```
GROQ_API_KEY=your_groq_api_key_here
```

---

## ⚡ Quick Start

```bash
# 1. Generate trend analysis
python src/generate_trends.py

# 2. Train RL agent (uses pre-computed backcasts)
python "RL Agent/pretrain_agent.py"

# 3. Generate unified RL forecast (2024–2040)
python "RL Agent/rl_forecast_unified.py"

# 4. Add uncertainty quantification
python "RL Agent/add_uncertainty.py"

# 5. Build stats database for RAG
python src/generate_stats_db.py

# 6. Launch Streamlit app
streamlit run src/app.py
```

Open http://localhost:8501 → Enter Groq API key → Ask questions!

### Generate PDF Report

```bash
python src/report_generator.py
# → Outputs climate_report.pdf
```

---

## ⚙️ Configuration

All parameters are centralized in `config.yaml`:

```yaml
forecasting:
  target_date: "2040-12-01"
  sarima:
    seasonal_period: 12
    max_p: 5
    max_q: 5
  lstm:
    look_back: 12
    epochs: 150
    batch_size: 16

drought:
  spi_window: 12
  spei_window: 12
  moderate_threshold: -1.0
  severe_threshold: -1.5

random_seeds:
  numpy: 42
  tensorflow: 42
  sklearn: 42
```

---

## 🧪 Testing

```bash
python test_pipeline.py
```

Validates: data loading, feature engineering (SPI, SPEI), trend analysis (Mann-Kendall), stationarity tests (ADF, KPSS), forecasting (SARIMA, Linear), and stats DB generation.

---

## 📚 Technology Stack

| Category          | Technologies                                            |
| ----------------- | ------------------------------------------------------- |
| **Core**          | Python 3.11, Pandas, NumPy, SciPy                       |
| **Statistical**   | statsmodels, pymannkendall, pmdarima                    |
| **Deep Learning** | TensorFlow / Keras (Bidirectional LSTM)                 |
| **ML**            | scikit-learn (Ridge, preprocessing)                     |
| **Time Series**   | Prophet                                                 |
| **RL**            | Custom Q-Learning (pickle persistence)                  |
| **RAG**           | ChromaDB, Sentence Transformers (e5-small-v2), Groq API |
| **Visualization** | Matplotlib, Seaborn                                     |
| **UI**            | Streamlit                                               |
| **Reports**       | FPDF                                                    |

---

## 📄 License

MIT License

---

## 🙏 Acknowledgments

- **Open-Meteo** — Historical weather data API
- **Groq** — Free LLM inference API
- **ChromaDB** — Vector database
- **Sentence Transformers** — Local embeddings
