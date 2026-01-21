import pandas as pd
import numpy as np
import pickle
from agent import RLAgent

def prepare_training_data():
    # 1. Load Data
    try:
        actuals = pd.read_csv('../Dataset/Algiers_Weather_Data.csv')
    except FileNotFoundError:
        actuals = pd.read_csv('Dataset/Algiers_Weather_Data.csv')
        
    sarima = pd.read_csv('../Predictions/sarima_forecast_2040.csv')
    lstm = pd.read_csv('../Predictions/lstm_forecast_2040.csv')

    # Load Linear if available
    try:
        linear = pd.read_csv('../Predictions/linear_forecast_2040_final.csv')
        has_linear = True
    except FileNotFoundError:
        has_linear = False

    # 2. Process Dates
    # Actuals: 2002-2023
    actuals['time'] = pd.to_datetime(actuals['time'])
    monthly_actuals = actuals.set_index('time').resample('MS')['temperature_2m_mean (°C)'].mean().reset_index()
    monthly_actuals.columns = ['Date', 'Actual']

    # Forecasts
    sarima.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
    sarima['Date'] = pd.to_datetime(sarima['Date'])
    
    lstm['Date'] = pd.to_datetime(lstm['Date']) # Assuming Date column exists

    # 3. Merge for TRAINING Period (May 2019 - Dec 2019)
    # We only have SARIMA from May 2019, so that's our limit.
    df = pd.merge(monthly_actuals, sarima[['Date', 'Forecast']], on='Date', how='inner')
    df = df.rename(columns={'Forecast': 'SARIMA'})
    
    df = pd.merge(df, lstm[['Date', 'Forecast']], on='Date', how='inner')
    df = df.rename(columns={'Forecast': 'LSTM'})
    
    # Note: We do NOT merge Linear here because Linear forecast (starts 2023) 
    # does not overlap with Pre-Training Period (2019).
    # We will initialize the Linear Q-values later based on SARIMA/LSTM performance.

    # Filter for Pre-Training Range
    train_df = df[(df['Date'] >= '2019-05-01') & (df['Date'] < '2020-01-01')].copy()
    
    print(f"Training Data Points: {len(train_df)} months (Repeated 1000x)")
    return train_df

def pretrain():
    train_df = prepare_training_data()
    
    n_actions = 2
    # Check if linear file exists (even if not in training data)
    import os
    if os.path.exists("../Predictions/linear_forecast_2040_final.csv"):
        n_actions = 3
        print(f"Linear forecast found. Agent will have {n_actions} actions.")
    
    # Create a "Blank" Agent
    # We use aggressive learning here because we want it to MEMORIZE 2019
    agent = RLAgent(n_actions=n_actions, lr=0.5, gamma=0.9, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995)

    # TRAINING LOOP (1000 Episodes)
    # The agent lives through 2019 over and over again.
    n_episodes = 1000
    
    for e in range(n_episodes):
        state = train_df.iloc[0]['Date'].month - 1
        
        for i in range(len(train_df) - 1):
            # 1. Action
            # Force action to be 0 or 1 since we only have data for those
            if np.random.uniform(0, 1) < agent.epsilon:
                action = np.random.choice([0, 1])
            else:
                action = np.argmax(agent.q_table[state, :2]) # Only consider 0 and 1
            
            # 2. Observation
            row = train_df.iloc[i]
            
            if action == 0:
                chosen_pred = row['SARIMA']
            elif action == 1:
                chosen_pred = row['LSTM']
            else:
                 chosen_pred = row['SARIMA'] # Fallback
            
            actual = row['Actual']
            
            # 3. Reward (Negative Absolute Error)
            error = abs(actual - chosen_pred)
            reward = -error
            
            # 4. Next State
            next_state = train_df.iloc[i+1]['Date'].month - 1
            
            # 5. Learn
            # Decay epsilon manually inside agent manually called or just manual update?
            # Agent decays in decision_policy. We skipped it. Call it to decay.
            agent.epsilon = max(agent.epsilon_end, agent.epsilon * agent.epsilon_decay)
            
            agent.update(state, action, reward, next_state)
            state = next_state

    # Initialize Linear Q-values (Column 2) to average of SARIMA/LSTM
    if n_actions > 2:
        print("Initializing Linear Q-values to average of SARIMA/LSTM...")
        for s in range(12):
            agent.q_table[s, 2] = np.mean(agent.q_table[s, :2])

    # Save the "Smart" Agent
    print("Training Complete. Saving Q-Table...")
    agent.save_model("pretrained_q_table.pkl")
    
    # Peek at what it learned
    print("\n--- What did it learn? (Q-Table) ---")
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    header = "Month | SARIMA Value | LSTM Value "
    if n_actions > 2:
        header += "| Linear Value "
    header += "| Preference"
    print(header)
    
    for m in range(12):
        s_val = agent.q_table[m, 0]
        l_val = agent.q_table[m, 1]
        
        vals = [s_val, l_val]
        if n_actions > 2:
            lin_val = agent.q_table[m, 2]
            vals.append(lin_val)
            
        best_idx = np.argmax(vals)
        if best_idx == 0: pref = "SARIMA"
        elif best_idx == 1: pref = "LSTM"
        else: pref = "Linear"
        
        row_str = f"{months[m]}   | {s_val:6.2f}       | {l_val:6.2f}     "
        if n_actions > 2:
            row_str += f"| {lin_val:6.2f}       "
        row_str += f"| {pref}"
        
        print(row_str)

if __name__ == "__main__":
    pretrain()
