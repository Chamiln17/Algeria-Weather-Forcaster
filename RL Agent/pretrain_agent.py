import pandas as pd
import numpy as np
import pickle
import os
from agent import RLAgent

def pretrain_dual_variable():
    """
    Pre-train RL agent using BOTH temperature and ET0 performance
    Combined reward ensures model is good for BOTH variables
    """
    print("="*70)
    print("TRAINING RL AGENT ON DUAL-VARIABLE PERFORMANCE")
    print("="*70)
    
    # Load BOTH backcast files
    temp_file = 'historical_backcasts_temperature_2019_2023_real.csv'
    et0_file = 'historical_backcasts_et0_2019_2023_real.csv'
    
    if not os.path.exists(temp_file) or not os.path.exists(et0_file):
        print(f"\n❌ ERROR: Backcast files not found!")
        print(f"   Expected: {temp_file}")
        print(f"             {et0_file}")
        print(f"   Run 'backcast_generator.ipynb' in Google Colab first.")
        return
    
    temp_df = pd.read_csv(temp_file)
    et0_df = pd.read_csv(et0_file)
    
    temp_df['Date'] = pd.to_datetime(temp_df['Date'])
    et0_df['Date'] = pd.to_datetime(et0_df['Date'])
    
    print(f"\n📊 Training Data: {len(temp_df)} months")
    print(f"   Period: {temp_df['Date'].min().date()} to {temp_df['Date'].max().date()}")
    print(f"   Variables: Temperature AND ET0 (dual-variable training)")
    print(f"   Models: SARIMA, LSTM, Ridge, Prophet")
    
    # Show performance on each variable
    print(f"\n📈 Temperature Performance (MAE):")
    for model in ['SARIMA', 'LSTM', 'Ridge', 'Prophet']:
        mae = np.abs(temp_df['Actual'] - temp_df[model]).mean()
        print(f"   {model:8}: {mae:.3f}°C")
    
    print(f"\n📈 ET0 Performance (MAE):")
    for model in ['SARIMA', 'LSTM', 'Ridge', 'Prophet']:
        mae = np.abs(et0_df['Actual'] - et0_df[model]).mean()
        print(f"   {model:8}: {mae:.3f} mm")
    
    # Calculate combined performance
    print(f"\n📊 Combined Performance (Normalized):")
    temp_best = min([np.abs(temp_df['Actual'] - temp_df[m]).mean() for m in ['SARIMA', 'LSTM', 'Ridge', 'Prophet']])
    et0_best = min([np.abs(et0_df['Actual'] - et0_df[m]).mean() for m in ['SARIMA', 'LSTM', 'Ridge', 'Prophet']])
    
    combined_scores = {}
    for model in ['SARIMA', 'LSTM', 'Ridge', 'Prophet']:
        temp_mae = np.abs(temp_df['Actual'] - temp_df[model]).mean()
        et0_mae = np.abs(et0_df['Actual'] - et0_df[model]).mean()
        combined = ((temp_mae / temp_best) + (et0_mae / et0_best)) / 2
        combined_scores[model] = combined
        print(f"   {model:8}: {combined:.3f}x (1.0 = best)")
    
    # Create agent
    n_actions = 4
    model_names = ['SARIMA', 'LSTM', 'Ridge', 'Prophet']
    
    print(f"\n🤖 Initializing RL Agent...")
    print(f"   Actions: {n_actions}")
    print(f"   States: 12 (months)")
    print(f"   Reward: Combined temperature + ET0 error")
    
    agent = RLAgent(
        n_actions=n_actions,
        lr=0.5,
        gamma=0.9,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995
    )
    
    # TRAINING LOOP with DUAL-VARIABLE rewards
    n_episodes = 1000
    print(f"\n🔄 Training for {n_episodes} episodes...")
    print(f"   (Learning from BOTH temperature and ET0 simultaneously)\\n")
    
    episode_rewards = []
    
    for e in range(n_episodes):
        total_reward = 0
        state = temp_df.iloc[0]['Date'].month - 1
        
        for i in range(len(temp_df)):
            # Agent selects action
            action = agent.decision_policy(state)
            
            # Get predictions from selected model for BOTH variables
            temp_row = temp_df.iloc[i]
            et0_row = et0_df.iloc[i]
            
            if action == 0:
                temp_pred = temp_row['SARIMA']
                et0_pred = et0_row['SARIMA']
            elif action == 1:
                temp_pred = temp_row['LSTM']
                et0_pred = et0_row['LSTM']
            elif action == 2:
                temp_pred = temp_row['Ridge']
                et0_pred = et0_row['Ridge']
            elif action == 3:
                temp_pred = temp_row['Prophet']
                et0_pred = et0_row['Prophet']
            
            temp_actual = temp_row['Actual']
            et0_actual = et0_row['Actual']
            
            # Calculate COMBINED reward (normalized by scale)
            temp_error = abs(temp_actual - temp_pred) / temp_best
            et0_error = abs(et0_actual - et0_pred) / et0_best
            
            # Average normalized error (equal weight to both variables)
            combined_error = (temp_error + et0_error) / 2
            reward = -combined_error  # Negative error = reward
            total_reward += reward
            
            # Update Q-table
            if i < len(temp_df) - 1:
                next_state = temp_df.iloc[i + 1]['Date'].month - 1
                agent.update(state, action, reward, next_state)
                state = next_state
        
        episode_rewards.append(total_reward)
        
        if (e + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"   Episode {e+1:4d}/{n_episodes} | Avg Reward: {avg_reward:7.2f} | ε: {agent.epsilon:.4f}")
    
    # Save trained model
    print("\n💾 Saving trained Q-table...")
    agent.save_model("pretrained_q_table.pkl")
    print("✅ Saved to: pretrained_q_table.pkl")
    
    # Display learned preferences
    print("\n" + "="*80)
    print("📊 LEARNED MODEL PREFERENCES BY MONTH (Dual-Variable)")
    print("="*80)
    
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    header = "Month  "
    for model in model_names:
        header += f"| {model:8} "
    header += "| Best Model"
    print(header)
    print("-" * len(header))
    
    for m in range(12):
        row_str = f"{months[m]:6} "
        vals = agent.q_table[m, :n_actions]
        for val in vals:
            row_str += f"| {val:8.2f} "
        
        best_idx = np.argmax(vals)
        best_model = model_names[best_idx]
        row_str += f"| {best_model}"
        print(row_str)
    
    print("="*80)
    
    # Training summary
    print("\n📈 Training Summary:")
    print(f"   Episodes: {n_episodes}")
    print(f"   Final ε: {agent.epsilon:.4f}")
    print(f"   Avg Reward (last 100): {np.mean(episode_rewards[-100:]):.2f}")
    
    # Model selection distribution
    best_models = [model_names[np.argmax(agent.q_table[m, :n_actions])] for m in range(12)]
    
    print(f"\n📊 Model Selection Distribution:")
    from collections import Counter
    model_counts = Counter(best_models)
    for model, count in model_counts.most_common():
        percentage = (count / 12) * 100
        score = combined_scores.get(model, 0)
        print(f"   {model}: {count}/12 months ({percentage:.1f}%) - Combined Score: {score:.2f}x")
    
    print("\n✅ Training complete!")
    print("   Agent learned from BOTH temperature and ET0 performance")
    print("   Models with high error on either variable are penalized")
    print("\n   Now run: python rl_forecast_unified.py")

if __name__ == "__main__":
    pretrain_dual_variable()
