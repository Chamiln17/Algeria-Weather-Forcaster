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
    """Load all forecast CSVs and summarize"""
    forecasts = {}
    
    for csv_file in PREDICTIONS_DIR.glob('*_forecast_*.csv'):
        try:
            df = pd.read_csv(csv_file)
            model_name = csv_file.stem
            
            # Extract key statistics
            forecasts[model_name] = {
                'model': model_name,
                'forecast_period': f"{df['date'].min()} to {df['date'].max()}",
                'variables': list(df.columns),
                'summary_statistics': df.describe().to_dict(),
                'last_5_years': df.tail(60).to_dict('records')  # Last 5 years (60 months)
            }
            logger.info(f"Loaded forecast: {model_name}")
        except Exception as e:
            logger.error(f"Error loading {csv_file}: {e}")
    
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
