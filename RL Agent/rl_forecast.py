import pandas as pd
import numpy as np
import os
from agent import RLAgent

def generate_forecast():
    # 1. Load the Future Data (2024-2040)
    print("Loading forecast files...")
    df_sarima = pd.read_csv('../sarima_forecast_2040.csv')
    df_sarima.rename(columns={'Unnamed: 0': 'Date', 'Forecast': 'SARIMA'}, inplace=True)
    df_sarima['Date'] = pd.to_datetime(df_sarima['Date'])

    df_lstm = pd.read_csv('../lstm_forecast_2040.csv')
    df_lstm['Date'] = pd.to_datetime(df_lstm['Date'])
    df_lstm.rename(columns={'Forecast': 'LSTM'}, inplace=True)

    # Merge them so we have side-by-side values
    future_df = pd.merge(df_sarima[['Date', 'SARIMA']], df_lstm[['Date', 'LSTM']], on='Date', how='inner')
    
    # Filter for future only (2024-2040)
    future_df = future_df[(future_df['Date'] >= '2024-01-01') & (future_df['Date'] <= '2040-12-01')].copy()
    print(f"Generating forecast for {len(future_df)} months (2024-2040)...")

    # 2. Load the TRAINED Agent
    # We set epsilon=0.0 because we want the best decision, no guessing.
    agent = RLAgent(n_actions=2, epsilon_start=0.0, epsilon_end=0.0, epsilon_decay=0.0)
    
    if os.path.exists("pretrained_q_table.pkl"):
        agent.load_model("pretrained_q_table.pkl")
        print("✅ Loaded Trained Agent.")
    else:
        raise FileNotFoundError("Could not find 'pretrained_q_table.pkl'. Train the agent first!")

    # 3. Let the Agent Decide
    final_forecasts = []
    choices = []

    for index, row in future_df.iterrows():
        # Get State (Month 0-11)
        state = row['Date'].month - 1
        
        # --- FIX IS HERE: Use 'decision_policy', not 'get_action' ---
        action = agent.decision_policy(state) # 0=SARIMA, 1=LSTM
        
        if action == 0:
            value = row['SARIMA']
            choice = "SARIMA"
        else:
            value = row['LSTM']
            choice = "LSTM"
            
        final_forecasts.append(value)
        choices.append(choice)

    # 4. Save Results
    future_df['RL_Best_Forecast'] = final_forecasts
    future_df['Model_Used'] = choices
    
    # Clean output
    output_df = future_df[['Date', 'SARIMA', 'LSTM', 'RL_Best_Forecast', 'Model_Used']]
    
    filename = "final_rl_forecast_2040.csv"
    output_df.to_csv(filename, index=False)
    
    print(f"\n✅ Success! Forecast saved to '{filename}'")
    print("\nPreview:")
    print(output_df.head())
    
    # Quick Check: How often did it pick SARIMA?
    sarima_count = output_df['Model_Used'].value_counts().get('SARIMA', 0)
    lstm_count = output_df['Model_Used'].value_counts().get('LSTM', 0)
    print(f"\nStats: The Agent picked SARIMA {sarima_count} times and LSTM {lstm_count} times.")

if __name__ == "__main__":
    generate_forecast()