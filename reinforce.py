# Spring 2025, 535514 Reinforcement Learning
# HW1: REINFORCE with baseline and GAE

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

# Define a tensorboard writer
writer = SummaryWriter("./tb_record_1")
        
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
        
        # 提取 state 與 action 的維度
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n if self.discrete else env.action_space.shape[0]
        self.hidden_size = 128
        
        # 定義共用層、動作層與價值層
        self.fc1 = nn.Linear(self.observation_dim, self.hidden_size)
        self.action_head = nn.Linear(self.hidden_size, self.action_dim)
        self.value_head = nn.Linear(self.hidden_size, 1)
        # 權重初始化 (使用 Xavier 初始化)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.action_head.weight)
        nn.init.xavier_uniform_(self.value_head.weight)
        
        # 儲存動作與回報的記憶區
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
        x = F.relu(self.fc1(state))
        action_logits = self.action_head(x)
        action_prob = F.softmax(action_logits, dim=-1)
        state_value = self.value_head(x)
        return action_prob, state_value


    def select_action(self, state):
        """
            Select the action given the current state
            - The input is the state, and the output is the action to apply 
            (based on the learned stochastic policy)
            TODO:
                1. Implement the forward pass for both the action and the state value
        """
        # 將 state 轉為 tensor，並加入 batch 維度
        state = torch.from_numpy(state).float().unsqueeze(0)
        action_prob, state_value = self.forward(state)
        m = Categorical(action_prob)
        action = m.sample()
        
        # 儲存 log_prob 與 state value 到動作記憶中
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

        # 從後往前計算累積折扣回報
        for r in self.rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float)
        # 選擇性標準化 (可提升學習穩定性)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        for (log_prob, value), R in zip(saved_actions, returns):
            advantage = R - value.detach().squeeze(-1)
            policy_losses.append(-log_prob * advantage)
            value_losses.append(F.mse_loss(value.squeeze(-1), torch.tensor([R], dtype=torch.float)))
        
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        return loss

    def clear_memory(self):
        # 重置 rewards 與動作記憶
        del self.rewards[:]
        del self.saved_actions[:]

class GAE:
    def __init__(self, gamma, lambda_, num_steps):
        self.gamma = gamma
        self.lambda_ = lambda_
        self.num_steps = num_steps  # 若設定 num_steps = None，則代表全批次計算

    def __call__(self, rewards, values, done):
        """
        Implement Generalized Advantage Estimation (GAE) for your value prediction
        TODO (1): Pass correct corresponding inputs (rewards, values, and done) into the function arguments
        TODO (2): Calculate the Generalized Advantage Estimation and return the obtained value
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
        TODO (1): In each episode, 
        1. run the policy till the end of the episode and keep the sampled trajectory
        2. update both the policy and the value network at the end of episode

        TODO (2): In each episode, 
        1. record all the value you aim to visualize on tensorboard (lr, reward, length, ...)
    """
    model = Policy()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    ewma_reward = 0
    
    for i_episode in count(1):
        # 使用新版 Gym 的 reset 方法，取得 state 與 info（info 可忽略）
        state, _ = env.reset()  
        ep_reward = 0
        t = 0
        
        # 每個 episode 最多跑 9999 步以避免無限循環
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
        print('Episode {}\tlength: {}\treward: {}\t ewma reward: {}'.format(i_episode, t, ep_reward, ewma_reward))

        # 使用 Tensorboard 記錄數值
        writer.add_scalar('Reward', ep_reward, i_episode)
        writer.add_scalar('Episode Length', t, i_episode)
        writer.add_scalar('EWMA Reward', ewma_reward, i_episode)

        # 當 EWMA reward 超過環境的 reward_threshold 則儲存模型並結束訓練
        if ewma_reward > env.spec.reward_threshold:
            if not os.path.isdir("./preTrained"):
                os.mkdir("./preTrained")
            torch.save(model.state_dict(), './preTrained/CartPole_{}.pth'.format(lr))
            print("Solved! Running reward is now {} and the last episode runs to {} time steps!".format(ewma_reward, t))
            break

        loss = model.calculate_loss()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.clear_memory()


def test(name, n_episodes=10):
    """
        Test the learned model (no change needed)
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
    # 為了重現性，設定隨機種子
    random_seed = 10  
    lr = 0.01
    env = gym.make('CartPole-v1', render_mode="rgb_array")
    torch.manual_seed(random_seed)  
    train(lr)
    test(f'CartPole_{lr}.pth')
