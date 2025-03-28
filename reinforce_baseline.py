# Spring 2025, 535514 Reinforcement Learning
# HW1: REINFORCE with baseline
"""
REINFORCE with Baseline solving LunarLander-v2
Author: Kevin H. Heieh
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
import torch.optim.lr_scheduler as Scheduler
from torch.utils.tensorboard import SummaryWriter

# Tuple for storing action information, including log probability and state value
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

# Define TensorBoard recorder
writer = SummaryWriter("./tb_record_baseline/second_run")
        
class Policy(nn.Module):
    """
        Implement both policy network and the value network in one model
        - Note that here we let the actor and value networks share the first layer
        - Feel free to change the architecture (e.g. number of hidden layers and the width of each hidden layer) as you like
        - Feel free to add any member variables/functions whenever needed
        TODO:
            1. Initialize the network (including the GAE parameters, shared layer(s), the action layer(s), and the value layer(s))
            2. Random weight initialization of each layer
    """
    def __init__(self):
        super(Policy, self).__init__()
        
        # Get the dimensions of state and action from the environment
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.observation_dim = env.observation_space.shape[0]  # LunarLander-v2 has 8 dimensions
        self.action_dim = env.action_space.n  # LunarLander-v2 has 4 dimensions
        self.hidden_size = 256  # Increase hidden layer size to enhance network expressiveness
        
        ########## YOUR CODE HERE (5~10 lines) ##########
        # Define shared feature extraction layers
        self.fc1 = nn.Linear(self.observation_dim, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Define policy network (action output layer)
        self.action_head = nn.Linear(self.hidden_size, self.action_dim)
        
        # Define value network (state value output layer) - serves as baseline function
        self.value_head = nn.Linear(self.hidden_size, 1)

        # Weight initialization (using Xavier initialization to improve deep network training)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.action_head.weight)
        nn.init.xavier_uniform_(self.value_head.weight)
        ########## END OF YOUR CODE ##########
        
        # Memory for storing actions and rewards
        self.saved_actions = []
        self.rewards = []

    def forward(self, state):
        """
            Forward pass of both policy and value networks
            - The input is the state, and the outputs are the corresponding 
              action probability distirbution and the state value
            TODO:
                1. Implement the forward pass for both the action and the state value
        """
        ########## YOUR CODE HERE (3~5 lines) ##########
        # Shared feature extraction layers
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        # Action probability calculation
        action_logits = self.action_head(x)
        action_prob = F.softmax(action_logits, dim=-1)
        
        # State value calculation (baseline)
        state_value = self.value_head(x)
        ########## END OF YOUR CODE ##########
        
        return action_prob, state_value

    def select_action(self, state):
        """
            Select the action given the current state
            - The input is the state, and the output is the action to apply 
            (based on the learned stochastic policy)
            TODO:
                1. Implement the forward pass for both the action and the state value
        """
        ########## YOUR CODE HERE (3~5 lines) ##########
        # Convert state to tensor and add batch dimension
        state = torch.from_numpy(state).float().unsqueeze(0)
        
        # Get action probability and state value
        action_prob, state_value = self.forward(state)
        
        # Create categorical distribution
        m = Categorical(action_prob)
        
        # Sample action from distribution
        action = m.sample()
        ########## END OF YOUR CODE ##########

        # Save action's log probability and state value to action memory
        self.saved_actions.append(SavedAction(m.log_prob(action), state_value))
        
        return action.item()

    def calculate_loss(self, gamma=0.9995):
        """
            Calculate the loss (= policy loss + value loss) to perform backprop later
            TODO:
                1. Calculate rewards-to-go required by REINFORCE with the help of self.rewards
                2. Calculate the policy loss using the policy gradient
                3. Calculate the value loss using either MSE loss or smooth L1 loss
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
        
        ########## YOUR CODE HERE (8-15 lines) ##########
        # Normalize returns (improves training stability)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        # Calculate policy loss and value loss
        for (log_prob, value), R in zip(saved_actions, returns):
            # Use state value as baseline, calculate advantage
            advantage = R - value.detach().squeeze(-1)
            
            # Policy loss: log probability multiplied by advantage
            policy_losses.append(-log_prob * advantage)
            
            # Value loss: difference between predicted value and actual return
            value_losses.append(F.mse_loss(value.squeeze(-1), torch.tensor([R], dtype=torch.float)))
        
        # Combine losses, weight of value loss can be adjusted
        loss = torch.stack(policy_losses).sum() + 0.5 * torch.stack(value_losses).sum()
        ########## END OF YOUR CODE ##########
        
        return loss

    def clear_memory(self):
        """
        Clear rewards and actions memory
        """
        del self.rewards[:]
        del self.saved_actions[:]

def train(lr=0.002):
    """
    Train model using SGD (via backpropagation)
    - Execute policy until episode ends, save sampled trajectories
    - Update policy and value networks at the end of the episode
    - Record values for visualization on TensorBoard
    """
    # Initialize policy model and optimizer
    model = Policy()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Add learning rate scheduler to help convergence
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.95)
    
    # EWMA reward for tracking learning progress
    ewma_reward = 0
    
    # Run training episodes
    for i_episode in count(1):
        # Reset environment and episode reward
        state, _ = env.reset()  
        ep_reward = 0
        t = 0
        
        # Run a complete episode
        while True:
            # Select action
            action = model.select_action(state)
            
            # Execute action
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated            
            
            # Save reward
            model.rewards.append(reward)
            ep_reward += reward
            t += 1
            
            # Episode end handling
            if done:
                break
            
        # Calculate loss and update network
        loss = model.calculate_loss()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        model.clear_memory()

        # Update EWMA reward and record results
        ewma_reward = 0.05 * ep_reward + (1 - 0.05) * ewma_reward

        # Print episode information every 100 episodes
        if i_episode % 100 == 0:
            print('Episode {}\tLength: {}\tReward: {:.2f}\tEWMA Reward: {:.2f}'.format(
                i_episode, t, ep_reward, ewma_reward))

        # Record values in TensorBoard
        writer.add_scalar('Reward/Episode', ep_reward, i_episode)
        writer.add_scalar('Episode_Length', t, i_episode)
        writer.add_scalar('EWMA_Reward', ewma_reward, i_episode)
        writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], i_episode)
        writer.add_scalar('Episode_Reward', ep_reward, i_episode)
        writer.add_scalar('Loss/Policy', loss.item(), i_episode)
        
        # Save model and complete training, LunarLander-v2 solving threshold is 200
        if ewma_reward > 200:
            if not os.path.isdir("./preTrained"):
                os.mkdir("./preTrained")
            torch.save(model.state_dict(), './preTrained/LunarLander_baseline_{}.pth'.format(lr))
            print("Solved! Running reward is {:.2f}, and the last episode ran {} timesteps!".format(ewma_reward, t))
            break

def test(name, n_episodes=10):
    """
    Test the learned model
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
        print('Episode {}\tReward: {:.2f}'.format(i_episode, running_reward))
    env.close()

if __name__ == '__main__':
    # Set random seed to ensure reproducibility
    random_seed = 10  
    lr = 0.002
    
    # Create LunarLander-v2 environment
    env = gym.make('LunarLander-v2', render_mode="rgb_array")
    torch.manual_seed(random_seed)
    env.reset(seed=random_seed)
    
    train(lr)
    test(f'LunarLander_baseline_{lr}.pth')