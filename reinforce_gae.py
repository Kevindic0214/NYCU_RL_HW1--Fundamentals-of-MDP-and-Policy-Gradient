"""
REINFORCE with GAE 解決 LunarLander-v2
作者：Kevin H. Hsieh
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

# 定義一個有用的元組
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

# 定義三種不同的λ值
LAMBDA_VALUES = [0.0, 0.5, 0.99]

class Policy(nn.Module):
    """
        在同一個模型中實作策略網路和價值網路
        - 注意：這裡我們讓演員網路和價值網路共享第一層
        - 我們增加了網路的寬度和深度以適應更複雜的LunarLander環境
    """
    def __init__(self, gae_lambda=0.95):
        super(Policy, self).__init__()
        
        # 提取狀態與動作的維度
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n if self.discrete else env.action_space.shape[0]
        self.hidden_size = 128  # 增加隱藏層寬度以提高模型容量
        
        # GAE參數
        self.gae_lambda = gae_lambda
        self.gamma = 0.99
        
        # 定義共用層、動作層與價值層
        # 使用更深的網路結構
        self.fc1 = nn.Linear(self.observation_dim, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.action_head = nn.Linear(self.hidden_size, self.action_dim)
        self.value_head = nn.Linear(self.hidden_size, 1)

        # 權重初始化（使用 Xavier 初始化）
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.action_head.weight)
        nn.init.xavier_uniform_(self.value_head.weight)
        
        # 儲存動作、狀態值與回報的記憶區
        self.saved_actions = []
        self.rewards = []
        self.values = []
        self.dones = []

    def forward(self, state):
        """
            策略網路和價值網路的前向傳播
            - 輸入是狀態，輸出是對應的動作機率分布和狀態價值
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_logits = self.action_head(x)
        action_prob = F.softmax(action_logits, dim=-1)
        state_value = self.value_head(x)
        return action_prob, state_value

    def select_action(self, state):
        """
            根據當前狀態選擇動作
            - 輸入是狀態，輸出是要執行的動作
            （基於學習到的隨機策略）
        """
        # 將狀態轉為張量，並加入批次維度
        state = torch.from_numpy(state).float().unsqueeze(0)
        action_prob, state_value = self.forward(state)
        m = Categorical(action_prob)
        action = m.sample()
        
        # 儲存 log_prob、狀態價值與完成標記到記憶中
        self.saved_actions.append(SavedAction(m.log_prob(action), state_value))
        self.values.append(state_value.item())
        return action.item()

    def calculate_gae(self):
        """
            計算廣義優勢估計(GAE)
            - 使用GAE來平衡偏差與方差之間的權衡
        """
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        dones = np.array(self.dones)
        
        # 計算delta和GAE
        gae = 0
        advantages = []
        
        # 從後往前計算GAE
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1 or dones[i]:
                next_value = 0
            else:
                next_value = values[i+1]
                
            # TD誤差: 獎勵 + 折扣 * 下一狀態價值 - 當前狀態價值
            delta = rewards[i] + self.gamma * next_value * (1 - dones[i]) - values[i]
            
            # GAE: 當前TD誤差 + 折扣 * lambda * 前一狀態的GAE值
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)
            
        return torch.FloatTensor(advantages)

    def calculate_loss(self):
        """
            計算損失（= 策略損失 + 價值損失）以進行反向傳播
        """
        saved_actions = self.saved_actions
        rewards = self.rewards
        
        # 計算GAE優勢
        advantages = self.calculate_gae()
        
        # 計算回報(與優勢估計無關，僅用於值函數學習)
        returns = []
        R = 0
        for r, done in zip(rewards[::-1], self.dones[::-1]):
            R = 0 if done else r + self.gamma * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns)
        
        # 標準化優勢，可提高訓練穩定性
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        policy_losses = []
        value_losses = []
        
        for (log_prob, value), R, advantage in zip(saved_actions, returns, advantages):
            # 使用GAE計算的優勢函數來更新策略
            policy_losses.append(-log_prob * advantage)
            
            # 使用實際回報更新價值網路
            value_losses.append(F.mse_loss(value.squeeze(-1), torch.tensor([R], dtype=torch.float)))
        
        # 合併損失函數
        loss = torch.stack(policy_losses).sum() + 0.5 * torch.stack(value_losses).sum()
        return loss

    def clear_memory(self):
        # 重置記憶區
        del self.rewards[:]
        del self.saved_actions[:]
        del self.values[:]
        del self.dones[:]

def train(lambda_value, lr=0.001, gamma=0.99):
    """
        使用 GAE 訓練 REINFORCE 模型
        - lambda_value: GAE的λ參數，控制偏差-方差權衡
        - lr: 學習率
        - gamma: 折扣因子
    """
    # 創建模型和優化器
    model = Policy(gae_lambda=lambda_value)
    model.gamma = gamma
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 記錄實驗結果
    writer = SummaryWriter(f"./tb_record_gae_lambda_{lambda_value}")
    
    running_reward = 0
    log_interval = 100
    episode_rewards = []
    
    for i_episode in count(1):
        state, _ = env.reset()  
        ep_reward = 0
        t = 0
        model.dones = []  # 重置回合開始時的完成標記
        
        # 執行一個回合
        while True:
            action = model.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated            
            model.rewards.append(reward)
            model.dones.append(done)  # 儲存完成標記
            ep_reward += reward
            t += 1
            
            if done:
                break
        
        episode_rewards.append(ep_reward)
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward if running_reward else ep_reward
        
        # 記錄到Tensorboard
        writer.add_scalar('回報', ep_reward, i_episode)
        writer.add_scalar('回合長度', t, i_episode)
        writer.add_scalar('指數加權平均回報', running_reward, i_episode)
        
        # 顯示訓練進度
        if i_episode % log_interval == 0:
            print(f'λ={lambda_value}，回合 {i_episode}\t平均回報: {np.mean(episode_rewards[-log_interval:]):.2f}\t長度: {t}')
            episode_rewards = []
            
        # 檢查是否達到解決條件
        if running_reward > env.spec.reward_threshold:
            print(f"已解決環境! λ={lambda_value}，運行回報為 {running_reward:.2f}，回合 {i_episode}")
            if not os.path.isdir("./preTrained"):
                os.mkdir("./preTrained")
            torch.save(model.state_dict(), f'./preTrained/LunarLander_lambda_{lambda_value}.pth')
            break
            
        # 如果超過1000回合仍未解決，結束訓練
        if i_episode >= 1000:
            print(f"達到最大回合數，λ={lambda_value}，最終運行回報為 {running_reward:.2f}")
            if not os.path.isdir("./preTrained"):
                os.mkdir("./preTrained")
            torch.save(model.state_dict(), f'./preTrained/LunarLander_lambda_{lambda_value}.pth')
            break
        
        # 計算損失並更新模型
        loss = model.calculate_loss()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.clear_memory()
    
    writer.close()
    return running_reward, i_episode

def test(model_path, render=True, n_episodes=3):
    """
        測試學習到的模型
    """     
    model = Policy()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    max_episode_len = 1000
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
        print(f'回合 {i_episode}\t回報: {ep_reward:.2f}')
    
    print(f'平均回報: {total_reward/n_episodes:.2f}')
    env.close()

def compare_lambdas():
    """
        比較不同λ值的GAE效果
    """
    results = {}
    
    for lambda_value in LAMBDA_VALUES:
        print(f"\n開始訓練 λ={lambda_value} 的模型...")
        final_reward, episodes = train(lambda_value)
        results[lambda_value] = {"reward": final_reward, "episodes": episodes}
    
    # 顯示比較結果
    print("\n===== 不同λ值的GAE效果比較 =====")
    print("λ值\t最終回報\t學習回合數")
    for lambda_value in LAMBDA_VALUES:
        print(f"{lambda_value}\t{results[lambda_value]['reward']:.2f}\t{results[lambda_value]['episodes']}")
    
    # 測試最佳模型
    best_lambda = max(results.keys(), key=lambda x: results[x]['reward'])
    print(f"\n測試λ={best_lambda}的最佳模型")
    test(f'./preTrained/LunarLander_lambda_{best_lambda}.pth')

if __name__ == '__main__':
    # 為了重現性，設定隨機種子
    random_seed = 10  
    torch.manual_seed(random_seed)
    
    # 建立環境
    env = gym.make('LunarLander-v2', render_mode="rgb_array")
    
    # 比較不同λ值
    compare_lambdas() 