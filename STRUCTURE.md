# Project Structure Guide

## 📁 Directory Organization

### Current Structure → Improved Structure

We're reorganizing for clarity and professional standards:

```
OLD (Legacy)                          NEW (Recommended)
─────────────────────────────────────────────────────────────
Preprocessing.ipynb                → notebooks/01_preprocessing.ipynb
script.ipynb                       → (archived or merged)
verify_phase1_phase2.ipynb         → notebooks/verify_phase1_phase2.ipynb

Dataset/                           → data/raw/
Preprocessed_dataset/              → data/processed/

Forecasting_Models/                → notebooks/
├── sarima_final.ipynb            → 03_sarima_forecast.ipynb
├── linear_final.ipynb            → 04_linear_forecast.ipynb
└── lstm_final.ipynb              → 05_lstm_forecast.ipynb

Models/                            → models/ (unchanged)
Predictions/                       → predictions/ (unchanged, lowercase)
Results/                           → results/ (unchanged, lowercase)

(new)                              → src/
                                       ├── config.py
                                       ├── utils.py
                                       ├── preprocessing.py
                                       ├── features.py
                                       ├── models.py
                                       └── evaluation.py
```

---

## 🎯 File Naming Conventions

### Notebooks
- **Prefix with numbers** for execution order: `01_`, `02_`, `03_`
- **Use snake_case**: `linear_forecast.ipynb` not `LinearForecast.ipynb`
- **Be descriptive**: `03_sarima_forecast.ipynb` not `model3.ipynb`

### Python Modules
- **snake_case** for all files: `config.py`, `preprocessing.py`
- **One clear purpose** per file
- **Group related functions** together

### Data Files
- **Version suffix**: `_v2`, `_v3` for iterations
- **Descriptive names**: `algiers_monthly_processed_v2.csv`
- **Date stamps** for outputs: `forecast_2040_final.csv`

### Model Files
- **Include model type**: `sarima_model_final.pkl`
- **Include metadata**: `lstm_scaler_final.pkl`
- **Consistent naming**: `{model}_{artifact}_{version}.{ext}`

---

## Recommended Notebook Organization

Each notebook should follow this structure:

```python
# 1. Title & Overview (Markdown)
# 2. Imports & Config
# 3. Load Data
# 4. Exploratory Analysis (if applicable)
# 5. Processing/Modeling
# 6. Evaluation
# 7. Visualization
# 8. Save Outputs
# 9. Summary Statistics
```

---

## Migration Steps (Optional)

If you want to reorganize now:

```bash
# Create new structure
mkdir -p data/raw data/processed notebooks src

# Move data files
mv Dataset/* data/raw/
mv Preprocessed_dataset/* data/processed/

# Rename notebooks with prefixes
cd Forecasting_Models
mv sarima_final.ipynb ../notebooks/03_sarima_forecast.ipynb
mv linear_final.ipynb ../notebooks/04_linear_forecast.ipynb
mv lstm_final.ipynb ../notebooks/05_lstm_forecast.ipynb

# Move verification notebook
mv ../verify_phase1_phase2.ipynb ../notebooks/verify_phase1_phase2.ipynb

# Clean up old folders (backup first!)
# rmdir Dataset Preprocessed_dataset Forecasting_Models
```

---

## Version Control Best Practices

### What to Track
✅ Code (`.py`, `.ipynb`)  
✅ Config files (`config.yaml`, `requirements.txt`)  
✅ Documentation (`.md`)  
✅ Final results (small CSVs, plots)

### What to Ignore (.gitignore)
❌ Large datasets (`data/raw/*.csv`)  
❌ Model binaries (unless small)  
❌ Notebook checkpoints (`.ipynb_checkpoints/`)  
❌ Virtual environments (`venv/`)  
❌ IDE files (`.vscode/`, `.idea/`)

---

## Results Organization

```
results/
├── figures/
│   ├── exploratory/
│   │   ├── temp_distribution.png
│   │   └── seasonal_patterns.png
│   ├── forecasts/
│   │   ├── sarima_forecast_plot.png
│   │   ├── linear_forecast_plot.png
│   │   └── lstm_forecast_plot.png
│   └── comparisons/
│       └── model_comparison.png
├── tables/
│   ├── trend_analysis.csv
│   └── model_metrics.csv
└── reports/
    └── final_summary.pdf
```

---

## 🔗 Path References in Code

Always use **relative paths** from notebook location:

```python
# ✅ Good
data = pd.read_csv('../data/processed/algiers_monthly_processed_v2.csv')
model.save('../models/lstm_model_final.h5')

# ❌ Avoid absolute paths
data = pd.read_csv('C:/Users/hp probook/Desktop/Project-MLA/data.csv')
```

Or use the config system:

```python
from src.config import get_path
data = pd.read_csv(get_path('processed_data'))
```

---

## 📦 Package Structure (Future)

If converting to a Python package later:

```
algiers_forecast/
├── __init__.py
├── data/
│   ├── __init__.py
│   ├── loader.py
│   └── preprocessor.py
├── models/
│   ├── __init__.py
│   ├── sarima.py
│   ├── linear.py
│   └── lstm.py
├── evaluation/
│   ├── __init__.py
│   └── metrics.py
└── utils/
    ├── __init__.py
    └── plotting.py
```

---

**Next Step**: Run `git status` to see current state, then commit the new structure! 🚀
