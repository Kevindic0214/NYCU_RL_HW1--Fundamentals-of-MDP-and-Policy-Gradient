# Spring 2025, 535514 Reinforcement Learning
# HW1: REINFORCE with baseline and GAE
"""
REINFORCE with Baseline solving CartPole-v1
Author: Kevin H. Heieh
Date: 2025/03/25
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
import torch.optim.lr_scheduler as Scheduler
from torch.utils.tensorboard import SummaryWriter

# Define a useful tuple (optional)
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

# Define Tensorboard writer
writer = SummaryWriter("./tb_record_1")
        
class Policy(nn.Module):
    """
        Implementing policy network and value network in the same model
        - Note: here we let the actor network and value network share the first layer
        - You're free to change the architecture (e.g., number of hidden layers and width of each hidden layer)
        - You're free to add any member variables/functions whenever needed
        TODO:
            1. Initialize the network (including GAE parameters, shared layer, action layer, and value layer)
            2. Random weight initialization for each layer
    """
    def __init__(self):
        super(Policy, self).__init__()
        
        # Extract state and action dimensions
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n if self.discrete else env.action_space.shape[0]
        self.hidden_size = 64
        
        # Define shared layer, action layer and value layer
        self.fc1 = nn.Linear(self.observation_dim, self.hidden_size)
        self.action_head = nn.Linear(self.hidden_size, self.action_dim)
        self.value_head = nn.Linear(self.hidden_size, 1)

        # Weight initialization (using Xavier initialization)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.action_head.weight)
        nn.init.xavier_uniform_(self.value_head.weight)
        
        # Memory for storing actions and returns
        self.saved_actions = []
        self.rewards = []

    def forward(self, state):
        """
            Forward pass of the policy network and value network
            - Input is the state, output is the corresponding action probability distribution and state value
            TODO:
                1. Implement forward pass for actions and state values
        """
        x = F.relu(self.fc1(state))
        action_logits = self.action_head(x)
        action_prob = F.softmax(action_logits, dim=-1)
        state_value = self.value_head(x)
        return action_prob, state_value


    def select_action(self, state):
        """
            Select action based on current state
            - Input is the state, output is the action to execute
            (based on the learned stochastic policy)
            TODO:
                1. Implement forward pass for actions and state values
        """
        # Convert state to tensor and add batch dimension
        state = torch.from_numpy(state).float().unsqueeze(0)
        action_prob, state_value = self.forward(state)
        m = Categorical(action_prob)
        action = m.sample()
        
        # Save log_prob and state value in action memory
        self.saved_actions.append(SavedAction(m.log_prob(action), state_value))
        return action.item()


    def calculate_loss(self, gamma=0.99):
        """
            Calculate loss (= policy loss + value loss) for backpropagation
            TODO:
                1. Calculate future returns needed for REINFORCE using self.rewards
                2. Calculate policy loss using policy gradient
                3. Calculate value loss using MSE or smooth L1 loss
        """
        R = 0
        saved_actions = self.saved_actions
        policy_losses = [] 
        value_losses = [] 
        returns = []

        # Calculate cumulative discounted returns from back to front
        for r in self.rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float)
        # Optional normalization (can improve learning stability)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        for (log_prob, value), R in zip(saved_actions, returns):
            advantage = R - value.detach().squeeze(-1)
            policy_losses.append(-log_prob * advantage)
            value_losses.append(F.mse_loss(value.squeeze(-1), torch.tensor([R], dtype=torch.float)))
        
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        return loss

    def clear_memory(self):
        # Reset rewards and action memory
        del self.rewards[:]
        del self.saved_actions[:]

class GAE:
    def __init__(self, gamma, lambda_, num_steps):
        self.gamma = gamma
        self.lambda_ = lambda_
        self.num_steps = num_steps  # If num_steps = None, it means full batch calculation

    def __call__(self, rewards, values, done):
        """
        Implement Generalized Advantage Estimation (GAE) for value prediction
        TODO (1): Pass the correct corresponding inputs (rewards, values, and done flag) to the function parameters
        TODO (2): Calculate generalized advantage estimation and return the resulting values
        """
        gae = 0
        advantages = []
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[i+1]
            delta = rewards[i] + self.gamma * next_value - values[i]
            gae = delta + self.gamma * self.lambda_ * gae
            advantages.insert(0, gae)
        return advantages

def train(lr=0.01):
    """
        Train the model using SGD (via backpropagation)
        TODO (1): For each episode,
        1. Execute policy until episode termination and save sampled trajectory
        2. At end of episode, update policy and value networks

        TODO (2): For each episode,
        1. Record all the values to be visualized in Tensorboard (learning rate, rewards, length, etc.)
    """
    model = Policy()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    ewma_reward = 0
    
    for i_episode in count(1):
        state, _ = env.reset()  
        ep_reward = 0
        t = 0
        
        # Run at most 9999 steps per episode to avoid infinite loops
        while True:
            action = model.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated            
            model.rewards.append(reward)
            ep_reward += reward
            t += 1
            if done:
                break
            
        ewma_reward = 0.05 * ep_reward + (1 - 0.05) * ewma_reward
        print('Episode {}\tLength: {}\tReward: {}\tEWMA Reward: {}'.format(i_episode, t, ep_reward, ewma_reward))

        # Record values for Tensorboard
        writer.add_scalar('Reward', ep_reward, i_episode)
        writer.add_scalar('Episode_Length', t, i_episode)
        writer.add_scalar('EWMA_Reward', ewma_reward, i_episode)

        # Save the model and finish training when reaching reward threshold
        if ewma_reward > env.spec.reward_threshold:
            if not os.path.isdir("./preTrained"):
                os.mkdir("./preTrained")
            torch.save(model.state_dict(), './preTrained/CartPole_{}.pth'.format(lr))
            print("Solved! The current running reward is {}, and the last episode runs to {} time steps!".format(ewma_reward, t))
            break

        loss = model.calculate_loss()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.clear_memory()


def test(name, n_episodes=10):
    """
        Test the learned model (no changes needed)
    """     
    model = Policy()
    
    model.load_state_dict(torch.load('./preTrained/{}'.format(name)))
    
    render = True
    max_episode_len = 10000
    
    for i_episode in range(1, n_episodes+1):
        state, _ = env.reset()
        running_reward = 0
        for t in range(max_episode_len+1):
            action = model.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            running_reward += reward
            if render:
                env.render()
            if done:
                break
        print('Episode {}\tReward: {}'.format(i_episode, running_reward))
    env.close()
    

if __name__ == '__main__':
    # Set random seed for reproducibility
    random_seed = 10  
    lr = 0.01
    env = gym.make('CartPole-v1', render_mode="rgb_array")
    torch.manual_seed(random_seed)  
    train(lr)
    test(f'CartPole_{lr}.pth')
