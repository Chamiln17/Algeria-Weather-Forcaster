"""
Generate stats_db.json - Aggregate climate data for RAG
"""
import json
import logging
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directories
RESULTS_DIR = Path('Results')
PREDICTIONS_DIR = Path('Predictions')
OUTPUT_FILE = RESULTS_DIR / 'stats_db.json'

def load_trends() -> dict:
    """Load trend analysis from trends.json"""
    trends_path = RESULTS_DIR / 'trends.json'
    if not trends_path.exists():
        logger.warning(f"Trends file not found: {trends_path}")
        return {}
    
    with open(trends_path, 'r') as f:
        return json.load(f)

def load_forecasts() -> dict:
    """Load RL Agent forecasts for both temperature and ET0"""
    forecasts = {}
    
    # Load RL Agent forecasts - NEW: separate files for temperature and ET0
    rl_temp_path = Path('RL Agent/final_rl_temperature_forecast_2040.csv')
    rl_et0_path = Path('RL Agent/final_rl_et0_forecast_2040.csv')
    
    # Temperature RL Forecast
    if rl_temp_path.exists():
        try:
            df = pd.read_csv(rl_temp_path)
            
            # Extract statistics
            rl_stats = df['RL_Best_Forecast'].describe().to_dict()
            model_usage = df['Model_Used'].value_counts().to_dict()
            
            forecasts['RL_Agent_Temperature'] = {
                'model': 'RL Agent - Temperature (Dual-Variable Trained)',
                'variable': 'temperature_2m_mean',
                'unit': '°C',
                'forecast_period': f"{df['Date'].min()} to {df['Date'].max()}",
                'available_models': ['SARIMA', 'LSTM', 'Ridge', 'Prophet'],
                'training_method': 'Dual-variable Q-learning (Temperature + ET0 combined rewards)',
                'summary_statistics': {
                    'RL_Best_Forecast': rl_stats,
                    'Model_Selection': model_usage,
                    'Total_Predictions': len(df)
                },
                'last_5_years': df.tail(60)[['Date', 'RL_Best_Forecast', 'Model_Used', 'SARIMA', 'LSTM', 'Ridge', 'Prophet']].to_dict('records'),
                'full_forecast': df[['Date', 'RL_Best_Forecast', 'Model_Used']].to_dict('records')
            }
            logger.info(f"✅ Loaded Temperature RL forecast: {len(df)} predictions")
            logger.info(f"   Model usage: {model_usage}")
        except Exception as e:
            logger.error(f"Error loading temperature RL forecast: {e}")
    else:
        logger.warning(f"Temperature RL forecast not found: {rl_temp_path}")
    
    # ET0 RL Forecast
    if rl_et0_path.exists():
        try:
            df = pd.read_csv(rl_et0_path)
            
            # Extract statistics
            rl_stats = df['RL_Best_Forecast'].describe().to_dict()
            model_usage = df['Model_Used'].value_counts().to_dict()
            
            forecasts['RL_Agent_ET0'] = {
                'model': 'RL Agent - ET0 (Dual-Variable Trained)',
                'variable': 'et0_fao_evapotranspiration',
                'unit': 'mm',
                'forecast_period': f"{df['Date'].min()} to {df['Date'].max()}",
                'available_models': ['SARIMA', 'LSTM', 'Ridge', 'Prophet'],
                'training_method': 'Dual-variable Q-learning (Temperature + ET0 combined rewards)',
                'summary_statistics': {
                    'RL_Best_Forecast': rl_stats,
                    'Model_Selection': model_usage,
                    'Total_Predictions': len(df)
                },
                'last_5_years': df.tail(60)[['Date', 'RL_Best_Forecast', 'Model_Used', 'SARIMA', 'LSTM', 'Ridge', 'Prophet']].to_dict('records'),
                'full_forecast': df[['Date', 'RL_Best_Forecast', 'Model_Used']].to_dict('records')
            }
            logger.info(f"✅ Loaded ET0 RL forecast: {len(df)} predictions")
            logger.info(f"   Model usage: {model_usage}")
        except Exception as e:
            logger.error(f"Error loading ET0 RL forecast: {e}")
    else:
        logger.warning(f"ET0 RL forecast not found: {rl_et0_path}")
    
    return forecasts

def generate_stats_db():
    """Generate comprehensive stats database for RAG"""
    logger.info("Generating stats_db.json...")
    
    # Load all data
    trends = load_trends()
    forecasts = load_forecasts()
    
    # Create structured database
    stats_db = {
        'metadata': {
            'generated_at': pd.Timestamp.now().isoformat(),
            'description': 'Climate change analysis statistics for Algeria',
            'data_sources': ['trends.json', 'forecast CSVs']
        },
        'trends': trends,
        'forecasts': forecasts,
        'summary': {
            'num_trend_variables': len(trends),
            'num_forecast_models': len(forecasts),
            'forecast_models': list(forecasts.keys())
        }
    }
    
    # Save to file
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(stats_db, f, indent=2)
    
    logger.info(f"✅ Generated {OUTPUT_FILE}")
    logger.info(f"   - {len(trends)} trend variables")
    logger.info(f"   - {len(forecasts)} forecast models")
    
    return stats_db

if __name__ == '__main__':
    generate_stats_db()
