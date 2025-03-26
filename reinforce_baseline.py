"""
REINFORCE with Baseline 解決 LunarLander-v2
作者：Kevin H. Heieh
日期：2025/03/25
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

# 檢查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用裝置: {device}")

# 定義 TensorBoard 記錄器
writer = SummaryWriter("./tb_record_lunar")

class PolicyNetwork(nn.Module):
    """策略網路（Actor）"""
    def __init__(self, state_dim=8, action_dim=4, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.head = nn.Linear(hidden_size, action_dim)
        
        # 初始化權重
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.head.weight)
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.head(x), dim=-1)

class ValueNetwork(nn.Module):
    """值網路（Baseline）"""
    def __init__(self, state_dim=8, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.head = nn.Linear(hidden_size, 1)
        
        # 初始化權重
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.head.weight)
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.head(x)

class REINFORCE:
    """帶基線的 REINFORCE 演算法"""
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, entropy_coef=0.01):
        self.policy_net = PolicyNetwork(state_dim, action_dim).to(device)
        self.value_net = ValueNetwork(state_dim).to(device)
        self.optimizer = optim.Adam([
            {'params': self.policy_net.parameters(), 'lr': lr},
            {'params': self.value_net.parameters(), 'lr': lr}
        ])
        self.gamma = gamma
        self.entropy_coef = entropy_coef  # 熵正則化係數
        self.saved_log_probs = []
        self.rewards = []
        self.entropies = []
        
    def select_action(self, state):
        """根據策略選擇動作"""
        state = torch.FloatTensor(state).to(device)
        probs = self.policy_net(state)
        dist = Categorical(probs)
        action = dist.sample()
        self.saved_log_probs.append(dist.log_prob(action))
        self.entropies.append(dist.entropy())  # 計算熵用於正則化
        return action.item()
    
    def update(self):
        """更新策略網路和值網路"""
        returns = []
        R = 0
        # 計算折扣報酬
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)  # 標準化
        
        # 計算值網路預測
        states_array = np.array(self.states, dtype=np.float32)
        states = torch.FloatTensor(states_array).to(device)
        values = self.value_net(states).squeeze()
        
        # 計算優勢
        advantages = returns - values.detach()
        
        # 策略損失（含熵正則化）
        policy_loss = -torch.stack(self.saved_log_probs).to(device) * advantages
        entropy_loss = -torch.stack(self.entropies).to(device).mean()  # 最大化熵
        policy_loss = policy_loss.mean() + self.entropy_coef * entropy_loss
        
        # 值函數損失
        value_loss = F.mse_loss(values, returns)
        
        # 合併損失並反向傳播
        self.optimizer.zero_grad()
        (policy_loss + value_loss).backward()
        self.optimizer.step()
        
        # 清空快取
        self.saved_log_probs = []
        self.rewards = []
        self.entropies = []
        return policy_loss.item(), value_loss.item()

def train(env, agent, max_episodes=5000):
    """訓練迴圈"""
    ewma_reward = 0
    for episode in range(max_episodes):
        state, _ = env.reset()
        episode_reward = 0
        states = []
        while True:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.rewards.append(reward)
            states.append(state)
            episode_reward += reward
            state = next_state
            if done:
                break
        
        # 更新網路
        agent.states = states  # 暫時儲存狀態序列用於值網路計算
        policy_loss, value_loss = agent.update()
        
        # 計算 EWMA 報酬
        ewma_reward = 0.05 * episode_reward + 0.95 * ewma_reward
        
        # 記錄 TensorBoard
        writer.add_scalar("Reward/Episode", episode_reward, episode)
        writer.add_scalar("Reward/EWMA", ewma_reward, episode)
        writer.add_scalar("Loss/Policy", policy_loss, episode)
        writer.add_scalar("Loss/Value", value_loss, episode)
        
        # 每 100 回合印出進度
        if episode % 100 == 0:
            print(f"回合 {episode}, 報酬: {episode_reward:.1f}, EWMA: {ewma_reward:.1f}")
        
        # 檢查是否解算環境
        if ewma_reward >= 200:
            torch.save(agent.policy_net.state_dict(), "./lunar_lander_solved.pth")
            print(f"在第 {episode} 回合解算完成！")
            break

if __name__ == "__main__":
    env = gym.make("LunarLander-v2", render_mode="rgb_array")
    agent = REINFORCE(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        lr=3e-4,
        gamma=0.99,
        entropy_coef=0.01
    )
    train(env, agent)
    env.close()