# Spring 2025, 535514 Reinforcement Learning
# HW1: REINFORCE with baseline and GAE
"""
REINFORCE with Baseline 解決 LunarLander-v2
作者：Kevin H. Heieh
日期：2025/03/25
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

# 定義一個有用的元組（選擇性）
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

# 定義 Tensorboard 寫入器
writer = SummaryWriter("./tb_record_1")
        
class Policy(nn.Module):
    """
        在同一個模型中實作策略網路和價值網路
        - 注意：這裡我們讓演員網路和價值網路共享第一層
        - 您可以自由更改架構（例如：隱藏層的數量和每個隱藏層的寬度）
        - 您可以自由添加任何需要的成員變數/函數
        TODO：
            1. 初始化網路（包括 GAE 參數、共享層、動作層和價值層）
            2. 對每一層進行隨機權重初始化
    """
    def __init__(self):
        super(Policy, self).__init__()
        
        # 提取狀態與動作的維度
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n if self.discrete else env.action_space.shape[0]
        self.hidden_size = 64
        
        # 定義共用層、動作層與價值層
        self.fc1 = nn.Linear(self.observation_dim, self.hidden_size)
        self.action_head = nn.Linear(self.hidden_size, self.action_dim)
        self.value_head = nn.Linear(self.hidden_size, 1)

        # 權重初始化（使用 Xavier 初始化）
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.action_head.weight)
        nn.init.xavier_uniform_(self.value_head.weight)
        
        # 儲存動作與回報的記憶區
        self.saved_actions = []
        self.rewards = []

    def forward(self, state):
        """
            策略網路和價值網路的前向傳播
            - 輸入是狀態，輸出是對應的動作機率分布和狀態價值
            TODO：
                1. 實作動作和狀態價值的前向傳播
        """
        x = F.relu(self.fc1(state))
        action_logits = self.action_head(x)
        action_prob = F.softmax(action_logits, dim=-1)
        state_value = self.value_head(x)
        return action_prob, state_value


    def select_action(self, state):
        """
            根據當前狀態選擇動作
            - 輸入是狀態，輸出是要執行的動作
            （基於學習到的隨機策略）
            TODO：
                1. 實作動作和狀態價值的前向傳播
        """
        # 將狀態轉為張量，並加入批次維度
        state = torch.from_numpy(state).float().unsqueeze(0)
        action_prob, state_value = self.forward(state)
        m = Categorical(action_prob)
        action = m.sample()
        
        # 儲存 log_prob 與狀態價值到動作記憶中
        self.saved_actions.append(SavedAction(m.log_prob(action), state_value))
        return action.item()


    def calculate_loss(self, gamma=0.99):
        """
            計算損失（= 策略損失 + 價值損失）以進行反向傳播
            TODO：
                1. 使用 self.rewards 計算 REINFORCE 所需的未來回報
                2. 使用策略梯度計算策略損失
                3. 使用均方誤差損失或平滑 L1 損失計算價值損失
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
        # 選擇性標準化（可提升學習穩定性）
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        for (log_prob, value), R in zip(saved_actions, returns):
            advantage = R - value.detach().squeeze(-1)
            policy_losses.append(-log_prob * advantage)
            value_losses.append(F.mse_loss(value.squeeze(-1), torch.tensor([R], dtype=torch.float)))
        
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        return loss

    def clear_memory(self):
        # 重置回報與動作記憶
        del self.rewards[:]
        del self.saved_actions[:]

class GAE:
    def __init__(self, gamma, lambda_, num_steps):
        self.gamma = gamma
        self.lambda_ = lambda_
        self.num_steps = num_steps  # 若設定 num_steps = None，則代表全批次計算

    def __call__(self, rewards, values, done):
        """
        實作廣義優勢估計（GAE）用於價值預測
        TODO (1)：將正確的對應輸入（回報、價值和完成標記）傳入函數參數
        TODO (2)：計算廣義優勢估計並返回所得的值
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
        使用 SGD（透過反向傳播）訓練模型
        TODO (1)：在每個回合中，
        1. 執行策略直到回合結束並保存採樣的軌跡
        2. 在回合結束時更新策略和價值網路

        TODO (2)：在每個回合中，
        1. 記錄所有要在 Tensorboard 上視覺化的值（學習率、回報、長度等）
    """
    model = Policy()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    ewma_reward = 0
    
    for i_episode in count(1):
        state, _ = env.reset()  
        ep_reward = 0
        t = 0
        
        # 每個回合最多跑 9999 步以避免無限循環
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
        print('回合 {}\t長度: {}\t回報: {}\t指數加權平均回報: {}'.format(i_episode, t, ep_reward, ewma_reward))

        # 使用 Tensorboard 記錄數值
        writer.add_scalar('回報', ep_reward, i_episode)
        writer.add_scalar('回合長度', t, i_episode)
        writer.add_scalar('指數加權平均回報', ewma_reward, i_episode)

        # 當指數加權平均回報超過環境的回報閾值則儲存模型並結束訓練
        if ewma_reward > env.spec.reward_threshold:
            if not os.path.isdir("./preTrained"):
                os.mkdir("./preTrained")
            torch.save(model.state_dict(), './preTrained/CartPole_{}.pth'.format(lr))
            print("已解決！當前運行回報為 {}，最後一個回合運行到 {} 個時間步！".format(ewma_reward, t))
            break

        loss = model.calculate_loss()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.clear_memory()


def test(name, n_episodes=10):
    """
        測試學習到的模型（無需更改）
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
        print('回合 {}\t回報: {}'.format(i_episode, running_reward))
    env.close()
    

if __name__ == '__main__':
    # 為了重現性，設定隨機種子
    random_seed = 10  
    lr = 0.01
    env = gym.make('CartPole-v1', render_mode="rgb_array")
    torch.manual_seed(random_seed)  
    train(lr)
    test(f'CartPole_{lr}.pth')
