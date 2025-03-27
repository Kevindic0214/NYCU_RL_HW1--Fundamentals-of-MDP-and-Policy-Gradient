"""
REINFORCE with GAE Optimized Version for LunarLander-v2
Optimizations:
1. Added entropy regularization to promote exploration
2. Vectorized loss calculation
3. Shared feature extraction followed by separate Actor and Critic branches
4. Accumulated multiple episodes for batch updates
5. Using learning rate scheduler
Author: Kevin H. Hsieh
Date: 2025/03/27
"""

import os
import gym
from itertools import count
from collections import namedtuple
import numpy as np
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

# Set computing device: use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device used:", device)

# Modify: Add entropy field in SavedAction
SavedAction = namedtuple('SavedAction', ['log_prob', 'value', 'entropy'])

# Define three different lambda values for comparison
LAMBDA_VALUES = [0.9, 0.95, 0.99]

class Policy(nn.Module):
    """
    Implementation of a network that contains both Actor and Critic,
    with shared feature extraction followed by separate hidden layers
    """
    def __init__(self, gae_lambda=0.95):
        super(Policy, self).__init__()
        
        # Extract dimensions of state and action (dependent on global variable env)
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n if self.discrete else env.action_space.shape[0]
        self.hidden_size = 128  # Increase hidden layer width to enhance model capacity
        
        # GAE parameters
        self.gae_lambda = gae_lambda
        self.gamma = 0.995
        
        # Shared feature extraction layers
        self.fc1 = nn.Linear(self.observation_dim, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Actor branch
        self.actor_hidden = nn.Linear(self.hidden_size, self.hidden_size)
        self.action_head = nn.Linear(self.hidden_size, self.action_dim)
        
        # Critic branch
        self.critic_hidden = nn.Linear(self.hidden_size, self.hidden_size)
        self.value_head = nn.Linear(self.hidden_size, 1)
        
        # Weight initialization (using Xavier initialization)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.actor_hidden.weight)
        nn.init.xavier_uniform_(self.action_head.weight)
        nn.init.xavier_uniform_(self.critic_hidden.weight)
        nn.init.xavier_uniform_(self.value_head.weight)
        
        # Memory to store actions, rewards, and done flags
        self.saved_actions = []
        self.rewards = []
        self.dones = []
    
    def forward(self, state):
        """
        Input state, output action probability distribution and state value
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        # Actor branch
        actor_x = F.relu(self.actor_hidden(x))
        action_logits = self.action_head(actor_x)
        action_prob = F.softmax(action_logits, dim=-1)
        
        # Critic branch
        critic_x = F.relu(self.critic_hidden(x))
        state_value = self.value_head(critic_x)
        return action_prob, state_value
    
    def select_action(self, state):
        """
        Select action based on current state, and store log_prob, value, and entropy
        """
        # Convert state to tensor and move to device
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        action_prob, state_value = self.forward(state)
        m = Categorical(action_prob)
        action = m.sample()
        entropy = m.entropy()
        
        self.saved_actions.append(SavedAction(m.log_prob(action), state_value, entropy))
        return action.item()
    
    def calculate_gae(self):
        """
        Calculate advantages for each step using Generalized Advantage Estimation (GAE)
        """
        rewards = np.array(self.rewards)
        # Extract state values from saved_actions and convert to numpy array
        values = np.array([action.value.item() for action in self.saved_actions])
        dones = np.array(self.dones)
        
        gae = 0
        advantages = []
        # Calculate GAE from back to front
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1 or dones[i]:
                next_value = 0
            else:
                next_value = values[i+1]
            delta = rewards[i] + self.gamma * next_value * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        return torch.FloatTensor(advantages).to(device)
    
    def calculate_loss(self):
        """
        Vectorized calculation of policy and value losses, with entropy regularization
        """
        advantages = self.calculate_gae()  # shape: [T]
        
        # Calculate returns
        returns = []
        R = 0
        for r, done in zip(reversed(self.rewards), reversed(self.dones)):
            R = 0 if done else r + self.gamma * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns).to(device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        saved_log_probs = torch.stack([a.log_prob for a in self.saved_actions])
        saved_entropies = torch.stack([a.entropy for a in self.saved_actions])
        state_values = torch.cat([a.value for a in self.saved_actions]).squeeze(-1)
        
        # Policy loss with entropy regularization (coefficient can be adjusted as needed)
        policy_loss = -(saved_log_probs * advantages).sum() - 0.001 * saved_entropies.sum()
        value_loss = F.mse_loss(state_values, returns)
        loss = policy_loss + 0.7 * value_loss
        return loss
    
    def clear_memory(self):
        # Clear accumulated memories
        self.saved_actions = []
        self.rewards = []
        self.dones = []

def train(lambda_value, lr=0.001, gamma=0.995, batch_size=10):
    """
    Train REINFORCE model using GAE with batch updates
    """
    model = Policy(gae_lambda=lambda_value).to(device)
    model.gamma = gamma
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Set learning rate scheduler to multiply learning rate by 0.95 every 200 episodes
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.95)
    
    writer = SummaryWriter(f"./tb_record_gae_lambda_{lambda_value}")
    
    running_reward = 0
    log_interval = 100
    episode_rewards = []
    
    batch_count = 0  # Count episodes in the current batch
    
    for i_episode in count(1):
        state, _ = env.reset()
        ep_reward = 0
        t = 0
        
        while True:
            action = model.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            model.rewards.append(reward)
            model.dones.append(done)
            ep_reward += reward
            t += 1
            if done:
                break
        
        episode_rewards.append(ep_reward)
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward if running_reward else ep_reward
        
        writer.add_scalar('Reward', ep_reward, i_episode)
        writer.add_scalar('Episode Length', t, i_episode)
        writer.add_scalar('Running Average Reward', running_reward, i_episode)
        
        batch_count += 1  # Add one episode
        
        if i_episode % log_interval == 0:
            avg_reward = np.mean(episode_rewards[-log_interval:])
            print(f'λ={lambda_value}, Episode {i_episode}\tAverage reward: {avg_reward:.2f}\tLength: {t}')
        
        # Update model when accumulated batch_size episodes
        if batch_count == batch_size:
            loss = model.calculate_loss()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            model.clear_memory()
            batch_count = 0
        
        # Save model and end training when environment is solved
        if running_reward > env.spec.reward_threshold:
            print(f"Environment solved! λ={lambda_value}, Running reward: {running_reward:.2f}, Episode {i_episode}")
            if not os.path.isdir("./preTrained"):
                os.mkdir("./preTrained")
            torch.save(model.state_dict(), f'./preTrained/LunarLander_lambda_{lambda_value}.pth')
            break
        
        if i_episode >= 1000:
            print(f"Maximum episodes reached, λ={lambda_value}, Final running reward: {running_reward:.2f}")
            if not os.path.isdir("./preTrained"):
                os.mkdir("./preTrained")
            torch.save(model.state_dict(), f'./preTrained/LunarLander_lambda_{lambda_value}.pth')
            break
        
    writer.close()
    return running_reward, i_episode

def test(model_path, render=True, n_episodes=3):
    """
    Test the performance of trained model
    """
    model = Policy().to(device)
    # Use map_location to load model to correct device
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    max_episode_len = 5000
    total_reward = 0
    
    for i_episode in range(1, n_episodes+1):
        state, _ = env.reset()
        ep_reward = 0
        for t in range(max_episode_len+1):
            action = model.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            if render:
                env.render()
            if done:
                break
        total_reward += ep_reward
        print(f'Episode {i_episode}\tReward: {ep_reward:.2f}')
    
    avg_reward = total_reward / n_episodes
    print(f'Average reward: {avg_reward:.2f}')
    env.close()

def compare_lambdas():
    """
    Compare the effect of different lambda values in GAE,
    presenting results in terms of final reward and required episodes
    """
    results = {}
    for lambda_value in LAMBDA_VALUES:
        print(f"\nStarting training model with λ={lambda_value}...")
        final_reward, episodes = train(lambda_value)
        results[lambda_value] = {"reward": final_reward, "episodes": episodes}
    
    print("\n===== Comparison of GAE with different λ values =====")
    print("λ value\tFinal Reward\tTraining Episodes")
    for lambda_value in LAMBDA_VALUES:
        print(f"{lambda_value}\t{results[lambda_value]['reward']:.2f}\t{results[lambda_value]['episodes']}")
    
    best_lambda = max(results.keys(), key=lambda x: results[x]['reward'])
    print(f"\nTesting best model with λ={best_lambda}")
    test(f'./preTrained/LunarLander_lambda_{best_lambda}.pth')

if __name__ == '__main__':
    # Set random seed for reproducibility
    random_seed = 10  
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # Create environment
    env = gym.make('LunarLander-v2', render_mode="rgb_array")
    
    # Compare different λ values
    compare_lambdas()
