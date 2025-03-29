# Spring 2025, 535514 Reinforcement Learning
# HW1: REINFORCE with GAE
"""
REINFORCE with Generalized Advantage Estimation (GAE) solving LunarLander-v2
Modified from the original REINFORCE implementation
Author: Kevin H. Heieh (original author)
Date: 2025/03/28
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
import torch.optim.lr_scheduler as Scheduler
from torch.utils.tensorboard import SummaryWriter
import argparse

# Define a useful tuple for action log probabilities and values
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

# Parse command line arguments for lambda value
parser = argparse.ArgumentParser(description='REINFORCE with GAE')
parser.add_argument('--lambda_value', type=float, default=0.95, 
                    help='Lambda parameter for GAE (default: 0.95)')
args = parser.parse_args()

# Define Tensorboard writer with lambda value in the directory name
writer = SummaryWriter(f"./tb_record_gae_lambda_{args.lambda_value}")
        
class Policy(nn.Module):
    """
        Implement both policy network and the value network in one model
        - Note that here we let the actor and value networks share the first layer
        - Feel free to change the architecture (e.g. number of hidden layers and the width of each hidden layer) as you like
        - Feel free to add any member variables/functions whenever needed
        TODO:
            1. Initialize the network (including the GAE parameters, shared layer(s), the action layer(s), and the value layer(s))
            2. Random weight initialization of each layer
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
        
        # Extract state and action dimensions
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n if self.discrete else env.action_space.shape[0]
        
        # Larger hidden layers for LunarLander (more complex environment)
        self.hidden_size1 = 256
        self.hidden_size2 = 128
        
        ########## YOUR CODE HERE (5~10 lines) ##########
        # Define policy network layers with a deeper architecture for LunarLander
        # First shared layer
        self.fc1 = nn.Linear(self.observation_dim, self.hidden_size1)
        
        # Second shared layer
        self.fc2 = nn.Linear(self.hidden_size1, self.hidden_size2)
        
        # Actor (policy) and critic (value) heads
        self.action_head = nn.Linear(self.hidden_size2, self.action_dim)
        self.value_head = nn.Linear(self.hidden_size2, 1)

        # Weight initialization (using Xavier/Glorot initialization)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.action_head.weight)
        nn.init.xavier_uniform_(self.value_head.weight)
        
        # Convert model to double precision
        self.double()
        ########## END OF YOUR CODE ##########

        # Memory for storing trajectory information
        self.saved_actions = []
        self.rewards = []
        self.values = []  # Store values for GAE calculation
        self.dones = []   # Store episode termination flags for GAE calculation

    def forward(self, state):
        """
            Forward pass of both policy and value networks
            - The input is the state, and the outputs are the corresponding 
              action probability distirbution and the state value
            TODO:
                1. Implement the forward pass for both the action and the state value
        """
        ########## YOUR CODE HERE (3~5 lines) ##########
        # Shared layers with ReLU activation
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        # Action probability calculation
        action_logits = self.action_head(x)
        action_prob = F.softmax(action_logits, dim=-1)

        # State value calculation
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
        state = torch.from_numpy(state).double().unsqueeze(0)

        # Get action probability and state value
        action_prob, state_value = self.forward(state)

        # Sample an action from the action probability distribution
        m = Categorical(action_prob)
        action = m.sample()
        
        # Store value for GAE calculation
        self.values.append(state_value.item())
        ########## END OF YOUR CODE ##########
        
        # Save log_prob in action memory
        self.saved_actions.append(SavedAction(m.log_prob(action), state_value))

        return action.item()


    def calculate_loss(self, gamma=0.99, gae_lambda=0.95):
        """
            Calculate the loss (= policy loss + value loss) to perform backprop
            This version uses GAE for advantage estimation
            Args:
                gamma: discount factor (default 0.99 for LunarLander)
                gae_lambda: GAE lambda parameter
        """
        # Get saved data
        saved_actions = self.saved_actions
        policy_losses = []
        value_losses = []
        
        # Initialize GAE calculator
        gae = GAE(gamma, gae_lambda, num_steps=None)
        
        # Calculate advantages using GAE
        values = torch.tensor(self.values + [0.0], dtype=torch.double)  # Append 0 for last state
        rewards = torch.tensor(self.rewards, dtype=torch.double)
        dones = torch.tensor(self.dones, dtype=torch.double)
        
        advantages = gae(rewards, values, dones)
        advantages = torch.tensor(advantages, dtype=torch.double)
        
        # Normalize advantages (helps with training stability)
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)
        
        # Calculate returns for value target
        returns = advantages + torch.tensor(self.values, dtype=torch.double)
        
        # Calculate policy and value losses
        for (log_prob, value), R in zip(saved_actions, returns):
            # Policy loss: log probability weighted by advantage
            policy_losses.append(-log_prob * advantages[len(policy_losses)])
            
            # Value loss: smooth L1 loss between predicted value and return
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([R], dtype=torch.double)))
        
        # Combine losses
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        
        return loss


    def clear_memory(self):
        # Reset all saved trajectory information
        del self.rewards[:]
        del self.saved_actions[:]
        del self.values[:]
        del self.dones[:]

class GAE:
    def __init__(self, gamma, lambda_, num_steps):
        """
        Initialize GAE parameters
        Args:
            gamma: discount factor
            lambda_: GAE lambda parameter
            num_steps: number of steps for truncated GAE (None for full episode)
        """
        self.gamma = gamma
        self.lambda_ = lambda_
        self.num_steps = num_steps  # set num_steps = None to adapt full batch

    def __call__(self, rewards, values, done):
        """
            Implement Generalized Advantage Estimation (GAE) for value prediction
            
            Args:
                rewards: tensor of rewards for each step
                values: tensor of state values at each step (plus next state)
                done: tensor of episode termination flags
                
            Returns:
                advantages: calculated GAE advantages
        """
        ########## YOUR CODE HERE (8-15 lines) ##########
        advantages = torch.zeros_like(rewards)
        
        # Get episode length
        episode_length = len(rewards)
        
        # Initialize gae accumulator
        gae = 0
        
        # Calculate GAE from back to front (reverse order)
        for t in reversed(range(episode_length)):
            # If this is the last step or episode terminates, use reward only
            if t == episode_length - 1 or done[t]:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            # Calculate TD-error (delta)
            delta = rewards[t] + self.gamma * next_value * (1 - done[t]) - values[t]
            
            # Calculate GAE recursively
            # A_t = δ_t + γλA_{t+1}
            gae = delta + self.gamma * self.lambda_ * (1 - done[t]) * gae
            
            # Store calculated advantage
            advantages[t] = gae
        ########## END OF YOUR CODE ##########
        
        return advantages

def train(lr=0.002, gae_lambda=0.95):
    """
        Train the model using REINFORCE with GAE
        Args:
            lr: learning rate
            gae_lambda: GAE lambda parameter
    """
    # Initialize policy model
    model = Policy()
    
    # Use Adam optimizer with learning rate decay
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = Scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

    # EWMA reward for tracking the learning progress
    ewma_reward = 0
    
    # Run episodes
    for i_episode in count(1):
        # Reset the environment and episode reward
        state, _ = env.reset()  
        ep_reward = 0
        t = 0

        # Uncomment the following line to use learning rate scheduler
        # scheduler.step()
        
        # For each episode, only run 9999 steps to avoid entering infinite loop during the learning process
        
        # Step the scheduler
        scheduler.step()
        
        # Run episode until termination or truncation
        while True:
            # Select action based on current state
            action = model.select_action(state)
            
            # Apply action and observe next state and reward
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store reward and done flag
            model.rewards.append(reward)
            model.dones.append(done)
            
            # Update episode statistics
            ep_reward += reward
            t += 1
            
            # Update state
            state = next_state
            
            # Break if episode ends
            if done:
                break
        
        # Calculate loss using GAE and perform backpropagation
        loss = model.calculate_loss(gamma=0.99, gae_lambda=gae_lambda)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Clear model memory
        model.clear_memory()

        # Update EWMA reward and log the results
        ewma_reward = 0.05 * ep_reward + (1 - 0.05) * ewma_reward
        print(f'Episode {i_episode}\tLength: {t}\tReward: {ep_reward:.2f}\tEWMA Reward: {ewma_reward:.2f}')

        # Record values for Tensorboard visualization
        writer.add_scalar('Reward/Episode', ep_reward, i_episode)
        writer.add_scalar('Episode_Length', t, i_episode)
        writer.add_scalar('EWMA_Reward', ewma_reward, i_episode)
        writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], i_episode)
        writer.add_scalar('Loss/Total', loss.item(), i_episode)
        
        # Check if we have "solved" the LunarLander problem
        # The threshold is 200 for LunarLander-v2
        if ewma_reward > env.spec.reward_threshold:
            if not os.path.isdir("./preTrained"):
                os.mkdir("./preTrained")
            torch.save(model.state_dict(), 
                      f'./preTrained/LunarLander_gae_lambda_{gae_lambda}.pth')
            print(f"Solved! Running reward is now {ewma_reward:.2f} and "
                  f"the last episode runs to {t} time steps!")
            break
            
        # Also save periodically to track progress
        if i_episode % 100 == 0:
            if not os.path.isdir("./preTrained"):
                os.mkdir("./preTrained")
            torch.save(model.state_dict(), 
                      f'./preTrained/LunarLander_gae_lambda_{gae_lambda}_checkpoint.pth')


def test(name, n_episodes=10):
    """
        Test the learned model
    """     
    model = Policy()
    
    model.load_state_dict(torch.load(f'./preTrained/{name}'))
    
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
            running_reward += reward
            if render:
                 env.render()
                 env.render()
            if done:
                break
        print(f'Episode {i_episode}\tReward: {running_reward:.2f}')
    env.close()
    

if __name__ == '__main__':
    # For reproducibility, fix the random seed
    # For reproducibility, fix the random seed
    random_seed = 10  
    
    # Set up environment - LunarLander instead of CartPole
    env = gym.make('LunarLander-v2', render_mode="rgb_array")
    torch.manual_seed(random_seed)
    env.reset(seed=random_seed)
    
    # Train with the GAE lambda value from command line arguments
    train(lr=0.002, gae_lambda=args.lambda_value)
    
    # Test the trained model
    test(f'LunarLander_gae_lambda_{args.lambda_value}.pth')