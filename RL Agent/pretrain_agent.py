import pandas as pd
import numpy as np
import pickle
import os
from agent import RLAgent

def pretrain():
    """
    Pre-train RL agent using ACTUAL historical performance (2019-2023)
    This is the proper way - learning from real errors, not assumptions
    """
    print("="*70)
    print("TRAINING RL AGENT ON HISTORICAL PERFORMANCE")
    print("="*70)
    
    # Load historical backcasts (Temperature - primary training variable)
    backcast_file = 'historical_backcasts_temperature_2019_2023_real.csv'
    
    if not os.path.exists(backcast_file):
        # Fallback to old proxy file if real one doesn't exist
        backcast_file_old = 'historical_backcasts_2019_2023.csv'
        if os.path.exists(backcast_file_old):
            print(f"\n⚠️  WARNING: Using old proxy backcasts!")
            print(f"   For proper training, run 'backcast_generator.ipynb' in Colab")
            print(f"   and place real backcast files here\n")
            backcast_file = backcast_file_old
        else:
            print(f"\n❌ ERROR: No backcast file found!")
            print(f"   Expected: {backcast_file}")
            print(f"   Run 'backcast_generator.ipynb' in Google Colab first.")
            print(f"   Should generate: historical_backcasts_temperature_2019_2023_real.csv")
            print(f"                    historical_backcasts_et0_2019_2023_real.csv")
            return
    
    train_df = pd.read_csv(backcast_file)
    train_df['Date'] = pd.to_datetime(train_df['Date'])
    
    print(f"\n📊 Training Data: {len(train_df)} months")
    print(f"   Period: {train_df['Date'].min().date()} to {train_df['Date'].max().date()}")
    print(f"   Variable: Temperature (°C)")
    print(f"   Models: SARIMA, LSTM, Ridge, Prophet")

    
    print(f"\n📊 Training Data: {len(train_df)} months")
    print(f"   Period: {train_df['Date'].min().date()} to {train_df['Date'].max().date()}")
    print(f"   Models: SARIMA, LSTM, Ridge, Prophet")
    
    # Show actual performance on this data
    print("\n📈 Historical Model Performance (MAE):")
    for model in ['SARIMA', 'LSTM', 'Ridge', 'Prophet']:
        mae = (train_df['Actual'] - train_df[model]).abs().mean()
        print(f"   {model:8}: {mae:.3f}°C")
    
    # Create agent
    n_actions = 4  # SARIMA, LSTM, Ridge, Prophet
    model_names = ['SARIMA', 'LSTM', 'Ridge', 'Prophet']
    
    print(f"\n🤖 Initializing RL Agent...")
    print(f"   Actions: {n_actions}")
    print(f"   States: 12 (months)")
    
    agent = RLAgent(
        n_actions=n_actions,
        lr=0.5,              # High learning rate for fast convergence
        gamma=0.9,           # Consider future rewards
        epsilon_start=1.0,   # Start with full exploration
        epsilon_end=0.01,    # End with minimal exploration
        epsilon_decay=0.995  # Gradual decay
    )
    
    # TRAINING LOOP
    n_episodes = 1000
    print(f"\n🔄 Training for {n_episodes} episodes...")
    print("   (Replaying 2019-2023 to learn model preferences)\n")
    
    episode_rewards = []
    
    for e in range(n_episodes):
        total_reward = 0
        
        # Reset to first month
        state = train_df.iloc[0]['Date'].month - 1
        
        # Loop through all months in training data
        for i in range(len(train_df)):
            row = train_df.iloc[i]
            
            # Agent selects which model to use
            action = agent.decision_policy(state)
            
            # Get prediction from selected model
            if action == 0:
                prediction = row['SARIMA']
                model_used = 'SARIMA'
            elif action == 1:
                prediction = row['LSTM']
                model_used = 'LSTM'
            elif action == 2:
                prediction = row['Ridge']
                model_used = 'Ridge'
            elif action == 3:
                prediction = row['Prophet']
                model_used = 'Prophet'
            
            actual = row['Actual']
            
            # Calculate reward (negative error - higher is better)
            error = abs(actual - prediction)
            reward = -error
            total_reward += reward
            
            # Get next state (next month)
            if i < len(train_df) - 1:
                next_state = train_df.iloc[i + 1]['Date'].month - 1
                agent.update(state, action, reward, next_state)
                state = next_state
        
        episode_rewards.append(total_reward)
        
        # Print progress
        if (e + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"   Episode {e+1:4d}/{n_episodes} | Avg Reward: {avg_reward:7.2f} | ε: {agent.epsilon:.4f}")
    
    # Save trained model
    print("\n💾 Saving trained Q-table...")
    agent.save_model("pretrained_q_table.pkl")
    print("✅ Saved to: pretrained_q_table.pkl")
    
    # Display learned preferences
    print("\n" + "="*80)
    print("📊 LEARNED MODEL PREFERENCES BY MONTH")
    print("="*80)
    
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Header
    header = "Month  "
    for model in model_names:
        header += f"| {model:8} "
    header += "| Best Model"
    print(header)
    print("-" * len(header))
    
    # Each month's learned values
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
        print(f"   {model}: {count}/12 months ({percentage:.1f}%)")
    
    print("\n✅ Training complete!")
    print("   Agent learned from ACTUAL 2019-2023 performance")
    print("   Now run: python rl_forecast_unified.py")

if __name__ == "__main__":
    pretrain()
