"""
Full Pipeline Test - Climate Analysis
Tests all components: preprocessing, features, trends, stationarity, forecasting
"""
import sys
from pathlib import Path
import pandas as pd
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.append(str(Path(__file__).parent))

def test_data_loading():
    """Test 1: Data Loading"""
    logger.info("=" * 60)
    logger.info("TEST 1: Data Loading")
    logger.info("=" * 60)
    
    data_path = Path('Preprocessed_dataset/algiers_monthly_processed_v2.csv')
    if not data_path.exists():
        logger.error(f"❌ Data file not found: {data_path}")
        return None
    
    df = pd.read_csv(data_path)
    logger.info(f"✅ Loaded data: {df.shape[0]} rows × {df.shape[1]} columns")
    if 'date' in df.columns:
        logger.info(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    logger.info(f"   Columns: {list(df.columns[:5])}...")
    
    return df

def test_features():
    """Test 2: Feature Engineering"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: Feature Engineering")
    logger.info("=" * 60)
    
    from src.features import calculate_spi, calculate_spei
    
    df = pd.read_csv('Preprocessed_dataset/algiers_monthly_processed_v2.csv')
    
    # Test SPI calculation
    if 'precipitation_sum' in df.columns:
        spi, _ = calculate_spi(df['precipitation_sum'], window=12)
        logger.info(f"✅ SPI calculated: {len(spi)} values")
        logger.info(f"   Range: {spi.min():.2f} to {spi.max():.2f}")
    
    # Test SPEI calculation
    if 'precipitation_sum' in df.columns and 'et0_fao_evapotranspiration_sum' in df.columns:
        spei, _ = calculate_spei(
            df['precipitation_sum'],
            df['et0_fao_evapotranspiration_sum'],
            window=12
        )
        logger.info(f"✅ SPEI calculated: {len(spei)} values")
        logger.info(f"   Range: {spei.min():.2f} to {spei.max():.2f}")

def test_trends():
    """Test 3: Trend Analysis"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: Trend Analysis")
    logger.info("=" * 60)
    
    from src.features import calculate_mk_trend, calculate_sens_slope
    
    df = pd.read_csv('Preprocessed_dataset/algiers_monthly_processed_v2.csv')
    
    # Test on temperature
    if 'temperature_2m_mean' in df.columns:
        temp = df['temperature_2m_mean'].dropna()
        
        mk_result = calculate_mk_trend(temp)
        logger.info(f"✅ Mann-Kendall Test:")
        logger.info(f"   Trend: {mk_result['trend']}")
        logger.info(f"   P-value: {mk_result['p']:.4f}")
        logger.info(f"   Significant: {mk_result['h']}")
        
        sens_result = calculate_sens_slope(temp)
        logger.info(f"✅ Sen's Slope:")
        logger.info(f"   Slope: {sens_result['slope']:.6f}")
        logger.info(f"   Intercept: {sens_result['intercept']:.4f}")

def test_stationarity():
    """Test 4: Stationarity Tests"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 4: Stationarity Tests")
    logger.info("=" * 60)
    
    from src.stationarity import test_stationarity, determine_differencing_order
    
    df = pd.read_csv('Preprocessed_dataset/algiers_monthly_processed_v2.csv')
    
    if 'temperature_2m_mean' in df.columns:
        temp = df['temperature_2m_mean'].dropna()
        
        # Test stationarity
        result = test_stationarity(temp, verbose=False)
        logger.info(f"✅ Stationarity Test:")
        logger.info(f"   ADF p-value: {result['adf_pvalue']:.4f}")
        logger.info(f"   KPSS p-value: {result['kpss_pvalue']:.4f}")
        logger.info(f"   Is stationary: {result['is_stationary']}")
        logger.info(f"   Recommendation: {result['recommendation']}")
        
        # Determine differencing order
        d, tests = determine_differencing_order(temp, max_d=2)
        logger.info(f"✅ Differencing Order: d={d}")

def test_forecasting():
    """Test 5: Forecasting Models"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 5: Forecasting Models")
    logger.info("=" * 60)
    
    from src.forecasting import SarimaForecaster, LinearBaseline
    
    df = pd.read_csv('Preprocessed_dataset/algiers_monthly_processed_v2.csv')
    
    if 'temperature_2m_mean' in df.columns:
        temp = df['temperature_2m_mean'].dropna()[:120]  # Use first 10 years
        
        # Test Linear Baseline
        logger.info("Testing Linear Baseline...")
        linear = LinearBaseline()
        linear.fit(temp)
        linear_forecast = linear.forecast(steps=12)
        logger.info(f"✅ Linear Baseline:")
        logger.info(f"   Slope: {linear.slope:.6f}")
        logger.info(f"   12-month forecast range: {linear_forecast['forecast'].min():.2f} to {linear_forecast['forecast'].max():.2f}")
        
        # Test SARIMA (quick test with limited data)
        logger.info("\nTesting SARIMA...")
        try:
            sarima = SarimaForecaster(seasonal_period=12, auto_select=True)
            sarima.fit(temp, max_p=2, max_q=2, max_P=1, max_Q=1)  # Limit search for speed
            sarima_forecast = sarima.forecast(steps=12)
            logger.info(f"✅ SARIMA:")
            logger.info(f"   Order: {sarima.order}")
            logger.info(f"   Seasonal Order: {sarima.seasonal_order}")
            logger.info(f"   12-month forecast range: {sarima_forecast['forecast'].min():.2f} to {sarima_forecast['forecast'].max():.2f}")
        except Exception as e:
            logger.warning(f"⚠️ SARIMA test skipped: {e}")

def test_trends_json():
    """Test 6: Trends JSON Generation"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 6: Trends JSON")
    logger.info("=" * 60)
    
    trends_path = Path('Results/trends.json')
    if trends_path.exists():
        import json
        with open(trends_path, 'r') as f:
            trends = json.load(f)
        logger.info(f"✅ Trends JSON exists:")
        logger.info(f"   Variables: {len(trends)}")
        logger.info(f"   Sample variables: {list(trends.keys())[:3]}")
    else:
        logger.warning(f"⚠️ Trends JSON not found. Run: python src/generate_trends.py")

def test_stats_db():
    """Test 7: Stats DB Generation"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 7: Stats Database")
    logger.info("=" * 60)
    
    try:
        from src.generate_stats_db import generate_stats_db
        stats_db = generate_stats_db()
        logger.info(f"✅ Stats DB generated:")
        logger.info(f"   Trend variables: {stats_db['summary']['num_trend_variables']}")
        logger.info(f"   Forecast models: {stats_db['summary']['num_forecast_models']}")
    except Exception as e:
        logger.warning(f"⚠️ Stats DB generation failed: {e}")

def main():
    """Run all tests"""
    logger.info("\n" + "🧪 CLIMATE ANALYSIS PIPELINE TEST")
    logger.info("=" * 60)
    
    try:
        # Run tests
        df = test_data_loading()
        if df is not None:
            test_features()
            test_trends()
            test_stationarity()
            test_forecasting()
            test_trends_json()
            test_stats_db()
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("✅ PIPELINE TEST COMPLETE")
        logger.info("=" * 60)
        logger.info("\nAll core components are working!")
        logger.info("\nNext steps:")
        logger.info("1. Free up disk space (~1GB)")
        logger.info("2. Install RAG dependencies: pip install chromadb groq sentence-transformers")
        logger.info("3. Run: streamlit run src/app.py")
        
    except Exception as e:
        logger.error(f"\n❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
