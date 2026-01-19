import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from agent import RLAgent
from simulate_env import prepare_environment, run_simulation

# 1. Setup
env_df = prepare_environment()

# Determine n_actions based on available columns in prepared environment
n_actions = 2
if 'Linear' in env_df.columns:
    n_actions = 3

# 2. Load the "Smart" Agent
agent = RLAgent(n_actions=n_actions, lr=0.2, gamma=0.9, epsilon_start=0.0, epsilon_end=0.0, epsilon_decay=0.0)
try:
    agent.load_model("pretrained_q_table.pkl")
    print(f"✅ Loaded Pre-trained Brain. (Actions: {n_actions})")
except:
    print("⚠️ Using blank brain (Results might be worse).")

# 3. Run
results = run_simulation(agent, env_df)

# Align Data for Plotting
matched_env = env_df.iloc[:len(results)]

# 4. create the Professional Plot
plt.figure(figsize=(14, 7))
sns.set_style("whitegrid")

# Plot 1: The Models
plt.plot(results['Date'], matched_env['SARIMA'], color='green', linestyle=':', label='SARIMA (Base)', alpha=0.6)
plt.plot(results['Date'], matched_env['LSTM'], color='blue', linestyle=':', label='LSTM (Base)', alpha=0.6)
if 'Linear' in matched_env.columns:
    plt.plot(results['Date'], matched_env['Linear'], color='magenta', linestyle=':', label='Linear (Base)', alpha=0.6)

plt.plot(results['Date'], results['Actual'], color='black', linewidth=1.5, label='Actual Temperature', alpha=0.8)

# Plot 2: The Agent (Thicker, Solid Line)
# We color the line red to show it's our AI
plt.plot(results['Date'], results['Prediction'], color='#D62728', linewidth=2.5, label='RL Agent (Ours)')

# Formatting
plt.title('Final Results: RL Agent vs Traditional Models (2020-2023)', fontsize=16)
plt.ylabel('Temperature (°C)', fontsize=12)
plt.xlabel('Date', fontsize=12)
plt.legend(loc='upper left', frameon=True, framealpha=0.9)

# Highlight the improvement
total_error = results['Error'].sum()
plt.text(results['Date'].iloc[0], 30, f"Total Error: {total_error:.2f}\n(Lower is Better)", 
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='red'), fontsize=12, color='red')

plt.tight_layout()
plt.savefig("final_agent_performance.png", dpi=300)
print("\n✅ Graph saved as 'final_agent_performance.png'")
plt.show()
