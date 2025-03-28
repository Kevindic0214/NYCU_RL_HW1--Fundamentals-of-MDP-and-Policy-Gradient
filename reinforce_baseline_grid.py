# Spring 2025, 535514 Reinforcement Learning
# HW1: REINFORCE with baseline
"""
REINFORCE with Baseline solving LunarLander-v2
Author: Kevin H. Heieh
Date: 2025/03/27
"""
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
import random
import os

# Setting random seeds to ensure reproducibility
random_seed = 10
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# TensorBoard record path
writer_dir = "./tb_record_lunar_baseline"

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim=8, action_dim=4, hidden_size=256):
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
    def __init__(self, state_dim=8, hidden_size=256):
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
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99, entropy_coef=0.01, hidden_size=256):
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_size)
        self.value_net = ValueNetwork(state_dim, hidden_size)
        self.optimizer = optim.Adam([
            {'params': self.policy_net.parameters(), 'lr': lr},
            {'params': self.value_net.parameters(), 'lr': lr}
        ])
        # Add learning rate scheduler, multiply learning rate by 0.9 every 200 episodes
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=200, gamma=0.9)
        self.gamma = gamma
        self.entropy_coef = entropy_coef  # Entropy regularization coefficient
        self.saved_log_probs = []
        self.rewards = []
        self.entropies = []
        self.states = []  # Used to store state sequences for value network computation
        
    def select_action(self, state):
        """Select action based on policy"""
        state = torch.FloatTensor(state)
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
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        states_array = np.array(self.states, dtype=np.float32)
        states = torch.FloatTensor(states_array)
        values = self.value_net(states).squeeze()
        
        advantages = returns - values.detach()
        
        # Policy loss (including entropy regularization)
        policy_loss = -torch.stack(self.saved_log_probs) * advantages
        entropy_loss = -torch.stack(self.entropies).mean()
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
    Training loop, keeps track of metrics via TensorBoard and print statements,
    and returns the best EWMA reward and the episode when the environment was solved (if applicable).
    """
    ewma_reward = 0
    best_ewma = -float('inf')
    solved_episode = max_episodes
    for episode in range(max_episodes):
        state, _ = env.reset()
        episode_reward = 0
        states = []
        t = 0  # Count steps per episode
        
        while True:
            t += 1
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
        
        # Update learning rate
        agent.scheduler.step()
        current_lr = agent.scheduler.get_last_lr()[0]
        
        ewma_reward = 0.05 * episode_reward + 0.95 * ewma_reward
        best_ewma = max(best_ewma, ewma_reward)
        
        # Record TensorBoard metrics
        writer.add_scalar("Reward/Episode", episode_reward, episode)
        writer.add_scalar("Reward/EWMA", ewma_reward, episode)
        writer.add_scalar("Loss/Policy", policy_loss, episode)
        writer.add_scalar("Loss/Value", value_loss, episode)
        writer.add_scalar("Episode_Length", t, episode)
        writer.add_scalar("Learning_Rate", current_lr, episode)
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Reward: {episode_reward:.1f}, EWMA: {ewma_reward:.1f}, Steps: {t}, Learning rate: {current_lr:.6f}")
        
        # Consider environment solved if EWMA reward reaches 200, save the model
        if ewma_reward >= 200:
            torch.save(agent.policy_net.state_dict(), "./lunar_lander_solved.pth")
            print(f"Environment solved at episode {episode}!")
            solved_episode = episode
            break
    return best_ewma, solved_episode

def test(model_path, num_episodes=10, render=True):
    """
    Test the performance of a trained model in the LunarLander environment
    
    Parameters:
    model_path (str): Path to the model weights file
    num_episodes (int): Number of test episodes
    render (bool): Whether to render the environment
    """
    # Create test environment
    render_mode = "human" if render else "rgb_array"
    env = gym.make("LunarLander-v2", render_mode=render_mode)
    env.reset(seed=random_seed)  # Use the same random seed for reproducibility
    
    # Initialize policy network
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    hidden_size = 256  # Use the same hidden layer size as in training
    
    policy_net = PolicyNetwork(state_dim, action_dim, hidden_size)
    
    # Load model weights
    try:
        policy_net.load_state_dict(torch.load(model_path))
        policy_net.eval()  # Set to evaluation mode
        print(f"Successfully loaded model: {model_path}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    # Run tests
    total_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Choose action according to policy
            state_tensor = torch.FloatTensor(state)
            with torch.no_grad():
                probs = policy_net(state_tensor)
                dist = Categorical(probs)
                action = dist.sample().item()
            
            # Execute action
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        
        total_rewards.append(episode_reward)
        print(f"Test episode {episode+1}/{num_episodes}, Reward: {episode_reward:.1f}")
    
    # Display test results
    avg_reward = sum(total_rewards) / num_episodes
    print(f"\nTesting completed! Average reward: {avg_reward:.1f}")
    print(f"Highest reward: {max(total_rewards):.1f}, Lowest reward: {min(total_rewards):.1f}")
    
    env.close()
    return total_rewards

if __name__ == "__main__":
    # Choose mode: "train" or "test"
    mode = "train"
    
    if mode == "train":
        # Use the latest hyperparameter combination
        lr = 0.002
        gamma = 0.99
        entropy_coef = 0.01
        hidden_size = 256
        
        print(f"Using parameters: lr={lr}, gamma={gamma}, entropy_coef={entropy_coef}, hidden_size={hidden_size}")
        
        # Create environment and TensorBoard recorder
        env = gym.make("LunarLander-v2", render_mode="rgb_array")
        env.reset(seed=random_seed)  # Set environment random seed
        writer = SummaryWriter(writer_dir)
        
        # Initialize agent
        agent = REINFORCE(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            lr=lr,
            gamma=gamma,
            entropy_coef=entropy_coef,
            hidden_size=hidden_size
        )
        
        # Start training
        best_ewma, solved_episode = train(env, agent, writer, max_episodes=5000)
        
        print(f"\nTraining completed!")
        print(f"Best EWMA: {best_ewma:.1f}")
        if solved_episode < 5000:
            print(f"Environment solved at episode {solved_episode}!")
        else:
            print("Environment not solved within maximum episodes.")
        
        env.close()
        writer.close()
    
    elif mode == "test":
        # Test trained model
        model_path = "./lunar_lander_solved.pth"
        test(model_path, num_episodes=10, render=True)