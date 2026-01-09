import pandas as pd
import numpy as np
import pickle
from agent import RLAgent

def prepare_training_data():
    # 1. Load Data
    actuals = pd.read_csv('../Algiers_Weather_Data.csv')
    sarima = pd.read_csv('../sarima_forecast_2040.csv')
    lstm = pd.read_csv('../lstm_forecast_2040.csv')

    # 2. Process Dates
    # Actuals: 2002-2023
    actuals['time'] = pd.to_datetime(actuals['time'])
    monthly_actuals = actuals.set_index('time').resample('MS')['temperature_2m_mean (°C)'].mean().reset_index()
    monthly_actuals.columns = ['Date', 'Actual']

    # Forecasts
    sarima.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
    sarima['Date'] = pd.to_datetime(sarima['Date'])
    
    lstm['Date'] = pd.to_datetime(lstm['Date'])

    # 3. Merge for TRAINING Period (May 2019 - Dec 2019)
    # We only have SARIMA from May 2019, so that's our limit.
    df = pd.merge(monthly_actuals, sarima[['Date', 'Forecast']], on='Date', how='inner')
    df = df.rename(columns={'Forecast': 'SARIMA'})
    
    df = pd.merge(df, lstm[['Date', 'Forecast']], on='Date', how='inner')
    df = df.rename(columns={'Forecast': 'LSTM'})

    # Filter for Pre-Training Range
    train_df = df[(df['Date'] >= '2019-05-01') & (df['Date'] < '2020-01-01')].copy()
    
    print(f"Training Data Points: {len(train_df)} months (Repeated 1000x)")
    return train_df

def pretrain():
    train_df = prepare_training_data()
    
    # Create a "Blank" Agent
    # We use aggressive learning here because we want it to MEMORIZE 2019
    agent = RLAgent(n_actions=2, lr=0.5, gamma=0.9, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995)

    # TRAINING LOOP (1000 Episodes)
    # The agent lives through 2019 over and over again.
    n_episodes = 1000
    
    for e in range(n_episodes):
        state = train_df.iloc[0]['Date'].month - 1
        
        for i in range(len(train_df) - 1):
            # 1. Action
            action = agent.decision_policy(state)
            
            # 2. Observation
            row = train_df.iloc[i]
            chosen_pred = row['SARIMA'] if action == 0 else row['LSTM']
            actual = row['Actual']
            
            # 3. Reward (Negative Absolute Error)
            error = abs(actual - chosen_pred)
            reward = -error
            
            # 4. Next State
            next_state = train_df.iloc[i+1]['Date'].month - 1
            
            # 5. Learn
            agent.update(state, action, reward, next_state)
            state = next_state

    # Save the "Smart" Agent
    print("Training Complete. Saving Q-Table...")
    agent.save_model("pretrained_q_table.pkl")
    
    # Peek at what it learned
    print("\n--- What did it learn? (Q-Table) ---")
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    print("Month | SARIMA Value | LSTM Value | Preference")
    for m in range(12):
        s_val = agent.q_table[m, 0]
        l_val = agent.q_table[m, 1]
        pref = "SARIMA" if s_val > l_val else "LSTM"
        print(f"{months[m]}   | {s_val:6.2f}       | {l_val:6.2f}     | {pref}")

if __name__ == "__main__":
    pretrain()