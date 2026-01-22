import pandas as pd
import numpy as np

# Load both backcast files
temp = pd.read_csv('historical_backcasts_temperature_2019_2023_real.csv')
et0 = pd.read_csv('historical_backcasts_et0_2019_2023_real.csv')

print("="*70)
print("TEMPERATURE MAE (2019-2023):")
print("="*70)
for model in ['SARIMA', 'LSTM', 'Ridge', 'Prophet']:
    mae = np.abs(temp['Actual'] - temp[model]).mean()
    print(f"  {model:8}: {mae:.3f}°C")

print("\n" + "="*70)
print("ET0 MAE (2019-2023):")
print("="*70)
for model in ['SARIMA', 'LSTM', 'Ridge', 'Prophet']:
    mae = np.abs(et0['Actual'] - et0[model]).mean()
    print(f"  {model:8}: {mae:.3f} mm")

print("\n" + "="*70)
print("COMBINED PERFORMANCE (Normalized Average):")
print("="*70)
# Normalize by best performance for each variable
temp_best = min([np.abs(temp['Actual'] - temp[model]).mean() for model in ['SARIMA', 'LSTM', 'Ridge', 'Prophet']])
et0_best = min([np.abs(et0['Actual'] - et0[model]).mean() for model in ['SARIMA', 'LSTM', 'Ridge', 'Prophet']])

for model in ['SARIMA', 'LSTM', 'Ridge', 'Prophet']:
    temp_mae = np.abs(temp['Actual'] - temp[model]).mean()
    et0_mae = np.abs(et0['Actual'] - et0[model]).mean()
    
    # Normalize: 1.0 = best, higher = worse
    temp_norm = temp_mae / temp_best
    et0_norm = et0_mae / et0_best
    combined = (temp_norm + et0_norm) / 2
    
    print(f"  {model:8}: {combined:.3f} (Temp: {temp_norm:.2f}x, ET0: {et0_norm:.2f}x)")
