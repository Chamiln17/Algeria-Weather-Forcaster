import pandas as pd
import numpy as np
import os
from agent import RLAgent

def generate_forecast_for_variable(variable_name='temperature'):
    """
    Generate RL-guided forecast for a specific variable (temperature or et0)
    
    Args:
        variable_name: 'temperature' or 'et0'
    """
    # 1. Load ALL Forecast Files for the specified variable
    print("="*60)
    print(f"Loading {variable_name.upper()} forecast files for 4 models...")
    print("="*60)
    
    # SARIMA
    df_sarima = pd.read_csv(f'../Predictions/sarima_{variable_name}_forecast_2040.csv')
    df_sarima['Date'] = pd.to_datetime(df_sarima['date'])
    df_sarima.rename(columns={'forecast': 'SARIMA'}, inplace=True)
    print(f"✅ SARIMA: {len(df_sarima)} predictions")

    # LSTM
    df_lstm = pd.read_csv(f'../Predictions/lstm_{variable_name}_forecast_2040.csv')
    df_lstm['Date'] = pd.to_datetime(df_lstm['date'])
    df_lstm.rename(columns={'forecast': 'LSTM'}, inplace=True)
    print(f"✅ LSTM: {len(df_lstm)} predictions")

    # Ridge
    try:
        df_ridge = pd.read_csv(f'../Predictions/ridge_{variable_name}_forecast_2040.csv')
        df_ridge['Date'] = pd.to_datetime(df_ridge['date'])
        df_ridge.rename(columns={'forecast': 'Ridge'}, inplace=True)
        print(f"✅ Ridge: {len(df_ridge)} predictions")
    except FileNotFoundError:
        print(f"⚠️ Warning: ridge_{variable_name}_forecast_2040.csv not found")
        df_ridge = pd.DataFrame()
    
    # Prophet
    try:
        df_prophet = pd.read_csv(f'../Predictions/prophet_{variable_name}_forecast_2040.csv')
        df_prophet['Date'] = pd.to_datetime(df_prophet['date'])
        df_prophet.rename(columns={'forecast': 'Prophet'}, inplace=True)
        print(f"✅ Prophet: {len(df_prophet)} predictions")
    except FileNotFoundError:
        print(f"⚠️ Warning: prophet_{variable_name}_forecast_2040.csv not found")
        df_prophet = pd.DataFrame()

    # 2. Merge All Models
    print("\n📊 Merging all forecasts...")
    future_df = pd.merge(df_sarima[['Date', 'SARIMA']], df_lstm[['Date', 'LSTM']], on='Date', how='inner')
    
    if not df_ridge.empty:
        future_df = pd.merge(future_df, df_ridge[['Date', 'Ridge']], on='Date', how='inner')
    
    if not df_prophet.empty:
        future_df = pd.merge(future_df, df_prophet[['Date', 'Prophet']], on='Date', how='inner')
    
    # Filter for future only (2024-2040)
    future_df = future_df[(future_df['Date'] >= '2024-01-01') & (future_df['Date'] <= '2040-12-01')].copy()
    print(f"✅ Merged dataset: {len(future_df)} months (2024-2040)")
    print(f"   Available models: {[col for col in future_df.columns if col != 'Date']}")

    # 3. Configure RL Agent for Available Models
    n_actions = 2  # Minimum: SARIMA, LSTM
    if 'Ridge' in future_df.columns:
        n_actions = 3
    if 'Prophet' in future_df.columns:
        n_actions = 4  # All 4 models available
        
    print(f"\n🤖 RL Agent configured for {n_actions} models")
    
    # Load agent (epsilon=0 for pure exploitation)
    agent = RLAgent(n_actions=n_actions, epsilon_start=0.0, epsilon_end=0.0, epsilon_decay=0.0)
    
    # Try to load pre-trained Q-table (variable-specific or general)
    q_table_files = [
        f"pretrained_q_table_{variable_name}.pkl",  # Variable-specific
        "pretrained_q_table.pkl"  # General (fallback)
    ]
    
    loaded = False
    for q_file in q_table_files:
        if os.path.exists(q_file):
            agent.load_model(q_file)
            print(f"✅ Loaded pre-trained Q-table from: {q_file}")
            loaded = True
            break
    
    if not loaded:
        print("⚠️ Warning: No pre-trained model found. Using initialized agent.")

    # 4. Generate Predictions with RL Agent
    print("\n🔮 Generating RL-guided forecasts...")
    final_forecasts = []
    choices = []
    model_names = ['SARIMA', 'LSTM', 'Ridge', 'Prophet'][:n_actions]

    for index, row in future_df.iterrows():
        # State = Month (0-11)
        state = row['Date'].month - 1
        
        # Agent Decision (selects best model for this month)
        action = agent.decision_policy(state)
        
        # Map action to forecast
        if action == 0:
            value = row['SARIMA']
            choice = "SARIMA"
        elif action == 1:
            value = row['LSTM']
            choice = "LSTM"
        elif action == 2 and 'Ridge' in row:
            value = row['Ridge']
            choice = "Ridge"
        elif action == 3 and 'Prophet' in row:
            value = row['Prophet']
            choice = "Prophet"
        else:
            # Fallback to SARIMA
            value = row['SARIMA']
            choice = "SARIMA (Fallback)"
            
        final_forecasts.append(value)
        choices.append(choice)

    # 5. Save Results
    future_df['RL_Best_Forecast'] = final_forecasts
    future_df['Model_Used'] = choices
    
    # Prepare output columns
    cols = ['Date', 'SARIMA', 'LSTM']
    if 'Ridge' in future_df.columns:
        cols.append('Ridge')
    if 'Prophet' in future_df.columns:
        cols.append('Prophet')
    cols.extend(['RL_Best_Forecast', 'Model_Used'])
    
    output_df = future_df[cols]
    
    filename = f"final_rl_{variable_name}_forecast_2040.csv"
    output_df.to_csv(filename, index=False)
    
    print(f"\n✅ Success! Forecast saved to '{filename}'")
    print(f"   Columns: {list(output_df.columns)}")
    print("\n📊 Preview:")
    print(output_df.head(10))
    
    # Model Selection Statistics
    print(f"\n📈 Model Selection Statistics for {variable_name.upper()}:")
    model_usage = output_df['Model_Used'].value_counts()
    print(model_usage)
    print(f"\nTotal predictions: {len(output_df)}")
    
    # Show percentage
    print("\n📊 Model Usage (%):")
    for model, count in model_usage.items():
        percentage = (count / len(output_df)) * 100
        print(f"   {model}: {percentage:.1f}%")
    
    return output_df

def generate_both_forecasts():
    """
    Generate RL forecasts for BOTH temperature and ET0
    """
    print("\n" + "="*70)
    print("🌍 GENERATING RL FORECASTS FOR BOTH VARIABLES")
    print("="*70 + "\n")
    
    # Generate Temperature forecast
    print("\n🌡️  TEMPERATURE FORECAST")
    print("-" * 70)
    temp_df = generate_forecast_for_variable('temperature')
    
    print("\n\n")
    
    # Generate ET0 forecast
    print("💧 ET0 FORECAST")
    print("-" * 70)
    et0_df = generate_forecast_for_variable('et0')
    
    print("\n" + "="*70)
    print("✅ BOTH FORECASTS GENERATED SUCCESSFULLY!")
    print("="*70)
    print("\n📁 Output files:")
    print("   - final_rl_temperature_forecast_2040.csv")
    print("   - final_rl_et0_forecast_2040.csv")
    
    return temp_df, et0_df

if __name__ == "__main__":
    # Generate forecasts for both variables
    generate_both_forecasts()
