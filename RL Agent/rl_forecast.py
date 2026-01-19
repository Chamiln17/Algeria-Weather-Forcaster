import pandas as pd
import numpy as np
import os
from agent import RLAgent

def generate_forecast():
    # 1. Load the Future Data (2024-2040)
    print("Loading forecast files...")
    df_sarima = pd.read_csv('../Predictions/sarima_forecast_2040.csv')
    df_sarima.rename(columns={'Unnamed: 0': 'Date', 'Forecast': 'SARIMA'}, inplace=True)
    df_sarima['Date'] = pd.to_datetime(df_sarima['Date'])

    df_lstm = pd.read_csv('../Predictions/lstm_forecast_2040.csv')
    df_lstm['Date'] = pd.to_datetime(df_lstm['Date'])
    df_lstm.rename(columns={'Forecast': 'LSTM'}, inplace=True)

    # Load Linear Forecast
    try:
        df_linear = pd.read_csv('../Predictions/linear_forecast_2040_final.csv')
        if 'date' in df_linear.columns:
            df_linear.rename(columns={'date': 'Date'}, inplace=True)
        if 'temperature_forecast' in df_linear.columns:
            df_linear.rename(columns={'temperature_forecast': 'Linear'}, inplace=True)
            
        df_linear['Date'] = pd.to_datetime(df_linear['Date'])
        # Depending on CSV format, column might be 'Forecast' already
        if 'Forecast' in df_linear.columns:
             df_linear.rename(columns={'Forecast': 'Linear'}, inplace=True)
        print("Loaded Linear forecast.")
    except FileNotFoundError:
        print("Warning: linear_forecast_2040_final.csv not found.")
        df_linear = pd.DataFrame()

    # Merge them so we have side-by-side values
    future_df = pd.merge(df_sarima[['Date', 'SARIMA']], df_lstm[['Date', 'LSTM']], on='Date', how='inner')
    
    if not df_linear.empty:
        future_df = pd.merge(future_df, df_linear[['Date', 'Linear']], on='Date', how='inner')
    
    # Filter for future only (2024-2040)
    future_df = future_df[(future_df['Date'] >= '2024-01-01') & (future_df['Date'] <= '2040-12-01')].copy()
    print(f"Generating forecast for {len(future_df)} months (2024-2040)...")

    # 2. Load the TRAINED Agent
    # Determine n_actions
    n_actions = 2
    if 'Linear' in future_df.columns:
        n_actions = 3
        
    # We set epsilon=0.0 because we want the best decision, no guessing.
    agent = RLAgent(n_actions=n_actions, epsilon_start=0.0, epsilon_end=0.0, epsilon_decay=0.0)
    
    if os.path.exists("pretrained_q_table.pkl"):
        agent.load_model("pretrained_q_table.pkl")
        print("✅ Loaded Trained Agent.")
    else:
        # If running purely for test without training, we might warn but proceed or fail.
        # Ideally we desire a trained agent.
        print("Warning: Could not find 'pretrained_q_table.pkl'. Using random/initialized agent.")

    # 3. Let the Agent Decide
    final_forecasts = []
    choices = []

    for index, row in future_df.iterrows():
        # Get State (Month 0-11)
        state = row['Date'].month - 1
        
        # Agent Decision
        action = agent.decision_policy(state) # 0=SARIMA, 1=LSTM, 2=Linear
        
        # Map to value
        if action == 0:
            value = row['SARIMA']
            choice = "SARIMA"
        elif action == 1:
            value = row['LSTM']
            choice = "LSTM"
        elif action == 2 and 'Linear' in row:
            value = row['Linear']
            choice = "Linear"
        else:
            # Fallback if somehow action 2 selected but no linear data (should be handled by n_actions, but safety)
            value = row['SARIMA']
            choice = "SARIMA (Fallback)"
            
        final_forecasts.append(value)
        choices.append(choice)

    # 4. Save Results
    future_df['RL_Best_Forecast'] = final_forecasts
    future_df['Model_Used'] = choices
    
    # Clean output
    cols = ['Date', 'SARIMA', 'LSTM']
    if 'Linear' in future_df.columns:
        cols.append('Linear')
    cols.extend(['RL_Best_Forecast', 'Model_Used'])
    
    output_df = future_df[cols]
    
    filename = "final_rl_forecast_2040.csv"
    output_df.to_csv(filename, index=False)
    
    print(f"\n✅ Success! Forecast saved to '{filename}'")
    print("\nPreview:")
    print(output_df.head())
    
    # Quick Check: How often did it pick each?
    print("\nStats:")
    print(output_df['Model_Used'].value_counts())

if __name__ == "__main__":
    generate_forecast()
