
import sys
import json
import pandas as pd
from pathlib import Path

# Add src to pythonpath
sys.path.append(str(Path(__file__).parent.parent))

from src.features import calculate_mk_trend, calculate_sens_slope
from src.preprocessing import aggregate_to_monthly

def main():
    print("Loading data...")
    # Try different potential filenames
    possible_files = [
        'Preprocessed_dataset/algiers_monthly_processed_v2.csv',
        'Preprocessed_dataset/algiers_monthly_processed.csv',
        'Preprocessed_dataset/monthly_climate.csv'
    ]
    
    df = None
    for f in possible_files:
        if Path(f).exists():
            print(f"Found {f}")
            df = pd.read_csv(f, parse_dates=['date'], index_col='date')
            break
            
    if df is None:
        print("Error: No preprocessed data found.")
        return

    trends = {}
    variables = ['temperature_mean', 'precipitation', 'et0', 'temperature_max', 'temperature_min']
    
    # Filter variables that exist in df
    variables = [v for v in variables if v in df.columns]

    print("Calculating trends...")
    for var in variables:
        print(f"  Analysing {var}...")
        try:
            series = df[var].dropna()
            mk_res = calculate_mk_trend(series)
            sens_res = calculate_sens_slope(series)
            
            trends[var] = {
                'mann_kendall': mk_res,
                'sens_slope': sens_res,
                'mean': float(series.mean()),
                'std': float(series.std())
            }
        except Exception as e:
            print(f"  Error Analysing {var}: {e}")

    output_path = Path('Results/trends.json')
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(trends, f, indent=4)
        
    print(f"Trends saved to {output_path}")

if __name__ == "__main__":
    main()
