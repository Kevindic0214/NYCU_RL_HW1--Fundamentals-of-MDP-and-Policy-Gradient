"""
REINFORCE with GAE 優化版 for LunarLander-v2
優化項目：
1. 加入熵正則化促進探索
2. 向量化損失計算
3. 共享特徵提取後，分離 Actor 與 Critic 分支
4. 累積多個回合進行批次更新
5. 使用學習率調整器 (LR Scheduler)
作者：Kevin H. Hsieh (原始版本) / 優化修改版由 ChatGPT 整理
日期：2025/03/27
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

# 設定運算設備：若有 GPU 則使用 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用設備：", device)

# 修改：在 SavedAction 中增加 entropy 欄位
SavedAction = namedtuple('SavedAction', ['log_prob', 'value', 'entropy'])

# 定義三種不同的λ值進行比較
LAMBDA_VALUES = [0.9, 0.95, 0.99]

class Policy(nn.Module):
    """
    實作同時包含 Actor 與 Critic 的網路，但在共享兩層特徵後分支出各自隱藏層
    """
    def __init__(self, gae_lambda=0.95):
        super(Policy, self).__init__()
        
        # 提取狀態與動作的維度（依賴全域環境變數 env）
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n if self.discrete else env.action_space.shape[0]
        self.hidden_size = 128  # 增加隱藏層寬度以提升模型容量
        
        # GAE參數
        self.gae_lambda = gae_lambda
        self.gamma = 0.995
        
        # 共享特徵提取層
        self.fc1 = nn.Linear(self.observation_dim, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Actor 分支
        self.actor_hidden = nn.Linear(self.hidden_size, self.hidden_size)
        self.action_head = nn.Linear(self.hidden_size, self.action_dim)
        
        # Critic 分支
        self.critic_hidden = nn.Linear(self.hidden_size, self.hidden_size)
        self.value_head = nn.Linear(self.hidden_size, 1)
        
        # 權重初始化（使用 Xavier 初始化）
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.actor_hidden.weight)
        nn.init.xavier_uniform_(self.action_head.weight)
        nn.init.xavier_uniform_(self.critic_hidden.weight)
        nn.init.xavier_uniform_(self.value_head.weight)
        
        # 儲存動作、回報、與完成標記的記憶區
        self.saved_actions = []
        self.rewards = []
        self.dones = []
    
    def forward(self, state):
        """
        輸入狀態，輸出動作機率分布與狀態價值
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        # Actor 分支
        actor_x = F.relu(self.actor_hidden(x))
        action_logits = self.action_head(actor_x)
        action_prob = F.softmax(action_logits, dim=-1)
        
        # Critic 分支
        critic_x = F.relu(self.critic_hidden(x))
        state_value = self.value_head(critic_x)
        return action_prob, state_value
    
    def select_action(self, state):
        """
        根據當前狀態選擇動作，並儲存 log_prob, value 與 entropy
        """
        # 將 state 轉成 tensor 並移至 device
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        action_prob, state_value = self.forward(state)
        m = Categorical(action_prob)
        action = m.sample()
        entropy = m.entropy()
        
        self.saved_actions.append(SavedAction(m.log_prob(action), state_value, entropy))
        return action.item()
    
    def calculate_gae(self):
        """
        使用廣義優勢估計 (GAE) 計算每一步的優勢
        """
        rewards = np.array(self.rewards)
        # 從儲存的 saved_actions 中提取狀態價值，並轉成 numpy 陣列
        values = np.array([action.value.item() for action in self.saved_actions])
        dones = np.array(self.dones)
        
        gae = 0
        advantages = []
        # 從後往前計算 GAE
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
        向量化計算策略與價值損失，並加入熵正則化項
        """
        advantages = self.calculate_gae()  # shape: [T]
        
        # 計算回報 (Returns)
        returns = []
        R = 0
        for r, done in zip(reversed(self.rewards), reversed(self.dones)):
            R = 0 if done else r + self.gamma * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns).to(device)
        
        # 優勢標準化處理
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        saved_log_probs = torch.stack([a.log_prob for a in self.saved_actions])
        saved_entropies = torch.stack([a.entropy for a in self.saved_actions])
        state_values = torch.cat([a.value for a in self.saved_actions]).squeeze(-1)
        
        # 策略損失：加入熵正則化 (係數可根據需求調整)
        policy_loss = -(saved_log_probs * advantages).sum() - 0.001 * saved_entropies.sum()
        value_loss = F.mse_loss(state_values, returns)
        loss = policy_loss + 0.7 * value_loss
        return loss
    
    def clear_memory(self):
        # 清除本次累積的記憶
        self.saved_actions = []
        self.rewards = []
        self.dones = []

def train(lambda_value, lr=0.001, gamma=0.995, batch_size=10):
    """
    使用 GAE 訓練 REINFORCE 模型，並進行批次更新
    """
    model = Policy(gae_lambda=lambda_value).to(device)
    model.gamma = gamma
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # 設定學習率調整器，每 200 回合後將學習率乘以 0.95
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.95)
    
    writer = SummaryWriter(f"./tb_record_gae_lambda_{lambda_value}")
    
    running_reward = 0
    log_interval = 100
    episode_rewards = []
    
    batch_count = 0  # 累計批次內的回合數
    
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
        
        writer.add_scalar('回報', ep_reward, i_episode)
        writer.add_scalar('回合長度', t, i_episode)
        writer.add_scalar('指數加權平均回報', running_reward, i_episode)
        
        batch_count += 1  # 累加一個回合
        
        if i_episode % log_interval == 0:
            avg_reward = np.mean(episode_rewards[-log_interval:])
            print(f'λ={lambda_value}，回合 {i_episode}\t平均回報: {avg_reward:.2f}\t長度: {t}')
        
        # 當累積到 batch_size 回合後，進行一次模型更新
        if batch_count == batch_size:
            loss = model.calculate_loss()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            model.clear_memory()
            batch_count = 0
        
        # 若達到環境解決條件則儲存模型並結束訓練
        if running_reward > env.spec.reward_threshold:
            print(f"已解決環境! λ={lambda_value}，運行回報為 {running_reward:.2f}，回合 {i_episode}")
            if not os.path.isdir("./preTrained"):
                os.mkdir("./preTrained")
            torch.save(model.state_dict(), f'./preTrained/LunarLander_lambda_{lambda_value}.pth')
            break
        
        if i_episode >= 1000:
            print(f"達到最大回合數，λ={lambda_value}，最終運行回報為 {running_reward:.2f}")
            if not os.path.isdir("./preTrained"):
                os.mkdir("./preTrained")
            torch.save(model.state_dict(), f'./preTrained/LunarLander_lambda_{lambda_value}.pth')
            break
        
    writer.close()
    return running_reward, i_episode

def test(model_path, render=True, n_episodes=3):
    """
    測試學習到的模型效能
    """
    model = Policy().to(device)
    # 使用 map_location 將模型載入至正確設備
    model.load_state_dict(torch.load(model_path, map_location=device))
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
    
    avg_reward = total_reward / n_episodes
    print(f'平均回報: {avg_reward:.2f}')
    env.close()

def compare_lambdas():
    """
    比較不同λ值的GAE效果，並以最終回報與所需回合數呈現結果
    """
    results = {}
    for lambda_value in LAMBDA_VALUES:
        print(f"\n開始訓練 λ={lambda_value} 的模型...")
        final_reward, episodes = train(lambda_value)
        results[lambda_value] = {"reward": final_reward, "episodes": episodes}
    
    print("\n===== 不同λ值的GAE效果比較 =====")
    print("λ值\t最終回報\t學習回合數")
    for lambda_value in LAMBDA_VALUES:
        print(f"{lambda_value}\t{results[lambda_value]['reward']:.2f}\t{results[lambda_value]['episodes']}")
    
    best_lambda = max(results.keys(), key=lambda x: results[x]['reward'])
    print(f"\n測試λ={best_lambda}的最佳模型")
    test(f'./preTrained/LunarLander_lambda_{best_lambda}.pth')

if __name__ == '__main__':
    # 為了重現性，設定隨機種子
    random_seed = 10  
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # 建立環境
    env = gym.make('LunarLander-v2', render_mode="rgb_array")
    
    # 比較不同λ值
    compare_lambdas()
