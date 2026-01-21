import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from agent import RLAgent
from simulate_env import prepare_environment, run_simulation

def analyze():
    # 1. Prepare Environment (Validation Data: 2020-2023)
    env_df = prepare_environment()

    # Determine n_actions
    n_actions = 2
    if 'Linear' in env_df.columns:
        n_actions = 3

    # 2. Initialize Agent for TESTING (Not Training)
    # We set epsilon to 0.0 because we want it to use the Pre-Trained knowledge, not guess.
    # We keep a small learning rate (0.1) so it can still adapt slightly to new trends.
    agent = RLAgent(
        n_actions=n_actions, 
        lr=0.1, 
        gamma=0.9, 
        epsilon_start=0.0,  # No random exploration
        epsilon_end=0.0, 
        epsilon_decay=0.0
    )

    # 3. Load the Pre-Trained Brain
    model_file = "pretrained_q_table.pkl"
    if os.path.exists(model_file):
        agent.load_model(model_file)
        print(f"\n✅ SUCCESS: Loaded pre-trained model from '{model_file}'")
        print("The agent will now apply its knowledge from 2019 to the 2020-2023 period.")
    else:
        print(f"\n⚠️ WARNING: '{model_file}' not found!")
        print("Did you run 'python pretrain_agent.py' first?")
        print("Running with a blank agent (Expect high error)...")

    # 4. Run Simulation
    results = run_simulation(agent, env_df)

    # --- DIAGNOSTICS & VISUALIZATION ---
    
    # Align data lengths (Simulation stops 1 step early)
    matched_env = env_df.iloc[:len(results)].copy()

    # Calculate Oracle (Best Possible Performance)
    matched_env['Error_SARIMA'] = abs(matched_env['Actual'] - matched_env['SARIMA'])
    matched_env['Error_LSTM'] = abs(matched_env['Actual'] - matched_env['LSTM'])
    
    # List of errors to consider for min
    error_cols = ['Error_SARIMA', 'Error_LSTM']
    
    if 'Linear' in matched_env.columns:
        matched_env['Error_Linear'] = abs(matched_env['Actual'] - matched_env['Linear'])
        error_cols.append('Error_Linear')

    matched_env['Min_Possible_Error'] = matched_env[error_cols].min(axis=1)

    oracle_score = matched_env['Min_Possible_Error'].sum()
    agent_score = results['Error'].sum()
    regret = agent_score - oracle_score

    print(f"\n--- FINAL DIAGNOSTIC REPORT ---")
    print(f"Agent Total Error:  {agent_score:.2f}")
    print(f"Oracle Total Error: {oracle_score:.2f} (Theoretical Minimum)")
    print(f"Regret:             {regret:.2f}")
    
    if regret < 20:
        print("✅ STATUS: EXCELLENT. The agent is making near-perfect decisions.")
    else:
        print("❌ STATUS: NEEDS IMPROVEMENT. The agent is still choosing the wrong models.")

    # Plotting
    plt.figure(figsize=(12, 6))
    
    # 1. Plot Actual
    plt.plot(results['Date'], results['Actual'], 'k--', label='Actual Temp', alpha=0.4)
    
    # 2. Plot Models
    plt.plot(results['Date'], matched_env['SARIMA'], 'g:', label='SARIMA (Base)', alpha=0.3)
    plt.plot(results['Date'], matched_env['LSTM'], 'b:', label='LSTM (Base)', alpha=0.3)
    if 'Linear' in matched_env.columns:
        plt.plot(results['Date'], matched_env['Linear'], 'm:', label='Linear (Base)', alpha=0.3)
    
    # 3. Plot Agent Choice
    # Color the line segments based on choice
    # (Complex plotting omitted for stability, showing simple Agent line)
    plt.plot(results['Date'], results['Prediction'], 'r-', linewidth=2.5, label='RL Agent (Ours)')

    plt.title(f'Final Performance: RL Agent vs Baselines\nTotal Error: {agent_score:.2f} (Oracle: {oracle_score:.2f})')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('analysis_plot.png')
    plt.show()

    # Show choices explicitly
    print("\nModel Selection Timeline:")
    print(results[['Date', 'Chosen_Model', 'Error']].head(10))
    print(results['Chosen_Model'].value_counts())

if __name__ == "__main__":
    analyze()
