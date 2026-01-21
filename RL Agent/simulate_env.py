import pandas as pd
from agent import RLAgent
import numpy as np
import matplotlib.pyplot as plt

# --- PART 1: Data Preparation (The Environment) ---

def prepare_environment():
    print("Loading and aligning data...")
    
    # 1. Load Ground Truth (Daily Data -> Monthly Mean)
    # File: Algiers_Weather_Data.csv
    try:
        df_daily = pd.read_csv('../Dataset/Algiers_Weather_Data.csv')
    except FileNotFoundError:
        # Fallback 
        df_daily = pd.read_csv('Dataset/Algiers_Weather_Data.csv')
        
    df_daily['time'] = pd.to_datetime(df_daily['time'])
    df_daily.set_index('time', inplace=True)
    
    # Resample to Monthly Mean to get 'Actual' temperature
    df_actual = df_daily['temperature_2m_mean (°C)'].resample('MS').mean()
    
    # 2. Load Forecasts
    # File: sarima_forecast_2040.csv
    df_sarima = pd.read_csv('../Predictions/sarima_forecast_2040.csv', parse_dates=[0], index_col=0)
    
    # File: lstm_forecast_2040.csv
    # Note: 'Date' column exists
    df_lstm = pd.read_csv('../Predictions/lstm_forecast_2040.csv', parse_dates=['Date'])
    df_lstm.set_index('Date', inplace=True)

    # File: linear_forecast_2040_final.csv
    try:
        df_linear = pd.read_csv('../Predictions/linear_forecast_2040_final.csv')
        if 'date' in df_linear.columns:
            df_linear.rename(columns={'date': 'Date'}, inplace=True)
        if 'temperature_forecast' in df_linear.columns:
            df_linear.rename(columns={'temperature_forecast': 'Forecast'}, inplace=True)
            
        df_linear['Date'] = pd.to_datetime(df_linear['Date'])
        df_linear.set_index('Date', inplace=True)
    except FileNotFoundError:
        print("Warning: linear_forecast_2040_final.csv not found, proceeding without it.")
        df_linear = pd.DataFrame()
    
    # 3. Create the Simulation DataFrame (Validation Period)
    # We define the Simulation Period: Jan 2020 -> Aug 2023 (End of Actuals)
    start_date = '2020-01-01'
    end_date = '2023-08-01'
    
    # Create the main dataframe with Actuals
    env_df = pd.DataFrame(df_actual[start_date:end_date])
    env_df.columns = ['Actual']
    
    # Align and merge forecasts
    # We use .reindex() to ensure we only get rows matching our validation dates
    env_df['SARIMA'] = df_sarima['Forecast'].reindex(env_df.index)
    env_df['LSTM'] = df_lstm['Forecast'].reindex(env_df.index)
    if not df_linear.empty:
        env_df['Linear'] = df_linear['Forecast'].reindex(env_df.index)
    
    # Drop rows where Actual, SARIMA, or LSTM are missing. 
    # We allow Linear to be missing (NaN) if it starts later.
    env_df.dropna(subset=['Actual', 'SARIMA', 'LSTM'], inplace=True)
    
    print(f"Environment prepared. Range: {env_df.index.min().date()} to {env_df.index.max().date()}")
    print(f"Total Simulation Steps: {len(env_df)}")
    if 'Linear' in env_df.columns:
        print("Linear model included (may have NaNs).")
    return env_df

# --- PART 2: The Simulation Loop ---

def run_simulation(agent, env_df):
    """
    Simulates the monthly requests.
    Args:
        agent: Instance of RLAgent (from Task 1)
        env_df: The DataFrame containing Actual, SARIMA, LSTM, Linear
    """
    print("\n--- Starting Simulation ---")
    
    total_reward = 0
    history = [] # To store results for analysis
    
    # Iterate through time steps (Jan 2020 -> Aug 2023)
    # We stop one month early because we need 'next_state'
    for i in range(len(env_df) - 1):
        
        # A. OBSERVE STATE
        current_date = env_df.index[i]
        state = current_date.month - 1  # 0=Jan, 11=Dec
        
        # B. AGENT DECISION
        action = agent.decision_policy(state) # 0=SARIMA, 1=LSTM, 2=Linear
        
        # Map action to column name
        model_map = {0: 'SARIMA', 1: 'LSTM', 2: 'Linear'}
        
        # Fallback Logic
        chosen_col = model_map.get(action, 'SARIMA')
        
        # Check if chosen column exists and is not NaN
        if chosen_col not in env_df.columns or pd.isna(env_df.iloc[i][chosen_col]):
            # Fallback to SARIMA (or LSTM if SARIMA missing, but we dropped those)
            # If Linear chosen but NaN, switch to SARIMA
            # Optional: We could penalize the agent? But for now just fallback.
            chosen_col = 'SARIMA'
            chosen_model = f"SARIMA (Fallback from {model_map.get(action)})"
        else:
            chosen_model = chosen_col
        
        # C. GET PREDICTION & ACTUAL
        pred = env_df.iloc[i][chosen_col]
        actual = env_df.iloc[i]['Actual']
        
        # D. CALCULATE REWARD (Negative Absolute Error)
        error = abs(actual - pred)
        reward = -error
        
        # E. UPDATE AGENT
        # Determine next state (Month of the next row)
        next_date = env_df.index[i+1]
        next_state = next_date.month - 1
        
        agent.update(state, action, reward, next_state)
        
        # F. LOGGING
        total_reward += reward
        history.append({
            'Date': current_date,
            'Actual': actual,
            'Chosen_Model': chosen_model,
            'Prediction': pred,
            'Error': error
        })

    print("--- Simulation Complete ---")
    print(f"Total Cumulative Reward (Negative Error): {total_reward:.2f}")
    
    return pd.DataFrame(history)

# --- EXECUTION BLOCK ---
if __name__ == "__main__":
    # 1. Prepare Environment (2020-2023)
    env_df = prepare_environment()

    # Determine n_actions based on available columns
    n_actions = 2
    if 'Linear' in env_df.columns:
        n_actions = 3

    # 2. Load the SMART Agent
    # Important: Set epsilon=0.1 (Low) because we want to EXPLOIT what we learned, not explore randomly.
    agent = RLAgent(n_actions=n_actions, epsilon_start=0.0, epsilon_end=0.0, epsilon_decay=0.0) 
    try:
        agent.load_model("pretrained_q_table.pkl")
        print(f"✅ Loaded Pre-trained Q-Table! (Actions: {n_actions})")
    except:
        print("⚠️ Warning: No pre-trained model found. Starting from scratch.")

    # 3. Run Simulation
    results = run_simulation(agent, env_df)
        
    # 4. Preview Results
    print("\nSimulation Sample:")
    print(results.head())
