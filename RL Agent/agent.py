import numpy as np
import pickle

class RLAgent:
    def __init__(self, n_states=12, n_actions=2, lr=0.1, gamma=0.9, 
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.95):
        self.q_table = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        
        # Epsilon Decay Parameters
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        self.n_actions = n_actions

    def decision_policy(self, state):
        # 1. Decay Epsilon (Reduce exploration over time)
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # 2. Epsilon-Greedy Logic
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.n_actions) # Explore
        else:
            return np.argmax(self.q_table[state])   # Exploit

    def update(self, state, action, reward, next_state):
        current_q = self.q_table[state, action]
        max_next_q = np.max(self.q_table[next_state])
        
        # Q-Learning Update
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state, action] = new_q

    def save_model(self, filename='q_table.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load_model(self, filename='q_table.pkl'):
        with open(filename, 'rb') as f:
            self.q_table = pickle.load(f)
