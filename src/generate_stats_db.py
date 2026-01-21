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
    """Load RL Agent forecast from RL Agent directory"""
    forecasts = {}
    
    # Load RL Agent forecast
    rl_forecast_path = Path('RL Agent/final_rl_forecast_2040.csv')
    
    if not rl_forecast_path.exists():
        logger.warning(f"RL Agent forecast not found: {rl_forecast_path}")
        return forecasts
    
    try:
        df = pd.read_csv(rl_forecast_path)
        
        # Focus on the RL_Best_Forecast column which is the agent's selection
        model_name = 'RL_Agent_Forecast'
        
        # Extract key statistics for RL_Best_Forecast
        rl_forecast_stats = df['RL_Best_Forecast'].describe().to_dict()
        
        # Get model selection statistics
        model_usage = df['Model_Used'].value_counts().to_dict()
        
        forecasts[model_name] = {
            'model': 'RL Agent (Adaptive Model Selection)',
            'forecast_period': f"{df['Date'].min()} to {df['Date'].max()}",
            'variables': ['RL_Best_Forecast', 'Model_Used'],
            'summary_statistics': {
                'RL_Best_Forecast': rl_forecast_stats,
                'Model_Selection_Count': model_usage
            },
            'last_5_years': df.tail(60)[['Date', 'RL_Best_Forecast', 'Model_Used']].to_dict('records'),
            'full_forecast': df[['Date', 'RL_Best_Forecast', 'Model_Used']].to_dict('records')
        }
        logger.info(f"✅ Loaded RL Agent forecast with {len(df)} predictions")
        logger.info(f"   Model usage: {model_usage}")
        
    except Exception as e:
        logger.error(f"Error loading RL Agent forecast: {e}")
    
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
