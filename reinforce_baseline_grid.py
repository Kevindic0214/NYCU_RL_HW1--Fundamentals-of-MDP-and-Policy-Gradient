import numpy as np
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import itertools

# Check if GPU is available
device = torch.device("cpu")
print(f"Using device: {device}")

# Base directory for TensorBoard records
base_writer_dir = "./tb_record_lunar_grid"

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim=8, action_dim=4, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.head = nn.Linear(hidden_size, action_dim)
        
        # Initialize weights
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.head.weight)
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.head(x), dim=-1)

class ValueNetwork(nn.Module):
    def __init__(self, state_dim=8, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.head = nn.Linear(hidden_size, 1)
        
        # Initialize weights
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.head.weight)
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.head(x)

class REINFORCE:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, entropy_coef=0.01, hidden_size=128):
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_size).to(device)
        self.value_net = ValueNetwork(state_dim, hidden_size).to(device)
        self.optimizer = optim.Adam([
            {'params': self.policy_net.parameters(), 'lr': lr},
            {'params': self.value_net.parameters(), 'lr': lr}
        ])
        self.gamma = gamma
        self.entropy_coef = entropy_coef  # Entropy regularization coefficient
        self.saved_log_probs = []
        self.rewards = []
        self.entropies = []
        self.states = []  # Used to store state sequences for value network computation
        
    def select_action(self, state):
        """Select action based on policy"""
        state = torch.FloatTensor(state).to(device)
        probs = self.policy_net(state)
        dist = Categorical(probs)
        action = dist.sample()
        self.saved_log_probs.append(dist.log_prob(action))
        self.entropies.append(dist.entropy())  # Entropy calculation for regularization
        return action.item()
    
    def update(self):
        """Update policy network and value network"""
        returns = []
        R = 0
        # Compute discounted rewards
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        states_array = np.array(self.states, dtype=np.float32)
        states = torch.FloatTensor(states_array).to(device)
        values = self.value_net(states).squeeze()
        
        advantages = returns - values.detach()
        
        # Policy loss (including entropy regularization)
        policy_loss = -torch.stack(self.saved_log_probs).to(device) * advantages
        entropy_loss = -torch.stack(self.entropies).to(device).mean()
        policy_loss = policy_loss.mean() + self.entropy_coef * entropy_loss
        
        # Value network loss (MSE)
        value_loss = F.mse_loss(values, returns)
        
        self.optimizer.zero_grad()
        (policy_loss + value_loss).backward()
        self.optimizer.step()
        
        # Clear temporary information
        self.saved_log_probs = []
        self.rewards = []
        self.entropies = []
        self.states = []
        return policy_loss.item(), value_loss.item()

def train(env, agent, writer, max_episodes=5000):
    """
    Training loop that keeps track of diagnostics via TensorBoard and print statements,
    returns the best EWMA reward and the episode when the environment was solved (if any).
    """
    ewma_reward = 0
    best_ewma = -float('inf')
    solved_episode = max_episodes
    for episode in range(max_episodes):
        state, _ = env.reset()
        episode_reward = 0
        states = []
        while True:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.rewards.append(reward)
            states.append(state)
            episode_reward += reward
            state = next_state
            if done:
                break
        
        agent.states = states
        policy_loss, value_loss = agent.update()
        
        ewma_reward = 0.05 * episode_reward + 0.95 * ewma_reward
        best_ewma = max(best_ewma, ewma_reward)
        
        # Record TensorBoard metrics
        writer.add_scalar("Reward/Episode", episode_reward, episode)
        writer.add_scalar("Reward/EWMA", ewma_reward, episode)
        writer.add_scalar("Loss/Policy", policy_loss, episode)
        writer.add_scalar("Loss/Value", value_loss, episode)
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Reward: {episode_reward:.1f}, EWMA: {ewma_reward:.1f}")
        
        # Save model if EWMA reward reaches 200 (environment considered solved)
        if ewma_reward >= 200:
            torch.save(agent.policy_net.state_dict(), "./lunar_lander_solved.pth")
            print(f"Environment solved at episode {episode}!")
            solved_episode = episode
            break
    return best_ewma, solved_episode

def grid_search():
    """
    Use Grid Search to automatically find the best hyperparameter combination,
    while keeping training diagnostics for each combination.
    """
    # Define hyperparameter candidates
    lr_list = [1e-3]
    gamma_list = [0.95, 0.99]
    entropy_coef_list = [0.0, 0.005, 0.01]
    hidden_size_list = [128, 256]
    
    results = []
    best_config = None
    best_reward = -float('inf')
    
    total_configs = len(lr_list) * len(gamma_list) * len(entropy_coef_list) * len(hidden_size_list)
    config_count = 0
    
    for lr, gamma, entropy_coef, hidden_size in itertools.product(lr_list, gamma_list, entropy_coef_list, hidden_size_list):
        config_count += 1
        print(f"\nStarting test combination {config_count}/{total_configs}: lr={lr}, gamma={gamma}, entropy_coef={entropy_coef}, hidden_size={hidden_size}")
        # Create a new environment and TensorBoard recorder for each hyperparameter set
        env = gym.make("LunarLander-v2", render_mode="rgb_array")
        writer = SummaryWriter(f"{base_writer_dir}/lr_{lr}_gamma_{gamma}_entropy_{entropy_coef}_hs_{hidden_size}")
        agent = REINFORCE(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            lr=lr,
            gamma=gamma,
            entropy_coef=entropy_coef,
            hidden_size=hidden_size
        )
        best_ewma, solved_episode = train(env, agent, writer, max_episodes=5000)
        print(f"Combination lr={lr}, gamma={gamma}, entropy_coef={entropy_coef}, hidden_size={hidden_size} Best EWMA: {best_ewma:.1f}, Solved Episode: {solved_episode}")
        results.append({
            "lr": lr,
            "gamma": gamma,
            "entropy_coef": entropy_coef,
            "hidden_size": hidden_size,
            "best_ewma": best_ewma,
            "solved_episode": solved_episode
        })
        if best_ewma > best_reward:
            best_reward = best_ewma
            best_config = (lr, gamma, entropy_coef, hidden_size)
        env.close()
        writer.close()
    
    print("\nGrid search completed!")
    print(f"Best parameter combination: lr={best_config[0]}, gamma={best_config[1]}, entropy_coef={best_config[2]}, hidden_size={best_config[3]}, Best EWMA: {best_reward:.1f}")
    return results

if __name__ == "__main__":
    grid_search()
