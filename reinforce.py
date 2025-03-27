# Spring 2025, 535514 Reinforcement Learning
# HW1: REINFORCE 
"""
Vanilla REINFORCE solving CartPole-v1
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

# Define a useful tuple for action log probabilities
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

# Define Tensorboard writer
writer = SummaryWriter("./tb_record_vanilla")
        
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
        
        # Extract state and action dimensions
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n if self.discrete else env.action_space.shape[0]
        self.hidden_size = 128
        
        ########## YOUR CODE HERE (5~10 lines) ##########
        # Define policy network layers
        self.fc1 = nn.Linear(self.observation_dim, self.hidden_size)
        self.action_head = nn.Linear(self.hidden_size, self.action_dim)
        self.value_head = nn.Linear(self.hidden_size, 1)

        # Weight initialization (using Xavier initialization)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.action_head.weight)
        nn.init.xavier_uniform_(self.value_head.weight)
        
        # Convert model to double precision
        self.double()
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
        x = F.relu(self.fc1(state))
        
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
        ########## END OF YOUR CODE ##########
        
        # Save log_prob in action memory
        self.saved_actions.append(SavedAction(m.log_prob(action), state_value))

        return action.item()


    def calculate_loss(self, gamma=0.999):
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
        returns = torch.tensor(returns, dtype=torch.double)
        
        ########## YOUR CODE HERE (8-15 lines) ##########
        # Normalize the returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        # Vanilla REINFORCE: directly use returns to weight policy gradients
        for (log_prob, value), R in zip(saved_actions, returns):
            # Calculate advantage using state value as baseline
            advantage = R - value.item()

            # Policy loss: log probability multiplied by advantage
            policy_losses.append(-log_prob * advantage)

            # Value loss: smooth L1 loss
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([R], dtype=torch.double)))
        
        # Combine all the losses
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        ########## END OF YOUR CODE ##########

        return loss

    def clear_memory(self):
        # Reset rewards and action memory
        del self.rewards[:]
        del self.saved_actions[:]

class GAE:
    def __init__(self, gamma, lambda_, num_steps):
        self.gamma = gamma
        self.lambda_ = lambda_
        self.num_steps = num_steps          # set num_steps = None to adapt full batch

    def __call__(self, rewards, values, done):
        pass
        """
            Implement Generalized Advantage Estimation (GAE) for your value prediction
            TODO (1): Pass correct corresponding inputs (rewards, values, and done) into the function arguments
            TODO (2): Calculate the Generalized Advantage Estimation and return the obtained value
        """

        ########## YOUR CODE HERE (8-15 lines) ##########



        
        ########## END OF YOUR CODE ##########

def train(lr=0.01):
    """
        Train the model using SGD (via backpropagation)
        TODO (1): In each episode, 
        1. run the policy till the end of the episode and keep the sampled trajectory
        2. update both the policy and the value network at the end of episode

        TODO (2): In each episode, 
        1. record all the value you aim to visualize on tensorboard (lr, reward, length, ...)
    """
    model = Policy()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Learning rate scheduler (optional)
    scheduler = Scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

    # EWMA reward for tracking the learning progress
    ewma_reward = 0
    
    # Run inifinitely many episodes
    for i_episode in count(1):
        # Reset the environment and episode reward
        state, _ = env.reset()  
        ep_reward = 0
        t = 0
        
        # Uncomment the following line to use learning rate scheduler
        # scheduler.step()
        
        # For each episode, only run 9999 steps to avoid entering infinite loop during the learning process
        
        ########## YOUR CODE HERE (10-15 lines) ##########
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
        
        # Calculate loss and perform backpropagation
        loss = model.calculate_loss()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        model.clear_memory()
        ########## END OF YOUR CODE ##########

        # Update EWMA reward and log the results
        ewma_reward = 0.05 * ep_reward + (1 - 0.05) * ewma_reward
        print('Episode {}\tLength: {}\tReward: {}\tEWMA Reward: {}'.format(i_episode, t, ep_reward, ewma_reward))

        #Try to use Tensorboard to record the behavior of your implementation 
        ########## YOUR CODE HERE (4-5 lines) ##########
        # Record values for Tensorboard visualization
        writer.add_scalar('Reward/Episode', ep_reward, i_episode)
        writer.add_scalar('Episode_Length', t, i_episode)
        writer.add_scalar('EWMA_Reward', ewma_reward, i_episode)
        writer.add_scalar('Episode_Reward', ep_reward, i_episode)
        writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[-1], i_episode)
        writer.add_scalar('Loss/Policy', loss.item(), i_episode)
        ########## END OF YOUR CODE ##########
        
        # Check if we have "solved" the cart pole problem, use 120 as the threshold in LunarLander-v2
        if ewma_reward > env.spec.reward_threshold:
            if not os.path.isdir("./preTrained"):
                os.mkdir("./preTrained")
            torch.save(model.state_dict(), './preTrained/CartPole_{}.pth'.format(lr))
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(ewma_reward, t))
            break


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
    # For reproducibility, fix the random seed
    random_seed = 10  
    lr = 0.01
    env = gym.make('CartPole-v0', render_mode="rgb_array")
    torch.manual_seed(random_seed)  
    train(lr)
    test(f'CartPole_{lr}.pth')