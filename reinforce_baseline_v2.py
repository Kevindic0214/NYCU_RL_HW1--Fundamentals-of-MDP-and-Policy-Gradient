"""
REINFORCE with Baseline 解決 LunarLander-v2 (優化版)
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用裝置: {device}")

writer = SummaryWriter("./tb_record_lunar_optimized")

class PolicyNetwork(nn.Module):
    """3層策略網路"""
    def __init__(self, state_dim=8, action_dim=4, hidden_size=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.head = nn.Linear(hidden_size, action_dim)
        self._init_weights()
        
    def _init_weights(self):
        for layer in [self.fc1, self.fc2, self.fc3, self.head]:
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))  # 正交初始化更適合RL
            nn.init.constant_(layer.bias, 0.0)
            
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return F.softmax(self.head(x), dim=-1)

class ValueNetwork(nn.Module):
    """深度值網路"""
    def __init__(self, state_dim=8, hidden_size=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.head = nn.Linear(hidden_size, 1)
        self._init_weights()
        
    def _init_weights(self):
        for layer in [self.fc1, self.fc2, self.fc3, self.head]:
            nn.init.orthogonal_(layer.weight, gain=1.0)
            nn.init.constant_(layer.bias, 0.0)
            
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.head(x)

class REINFORCE:
    def __init__(self, state_dim, action_dim, lr=1e-4, gamma=0.995, entropy_coef=0.1):
        self.policy_net = PolicyNetwork(state_dim, action_dim).to(device)
        self.value_net = ValueNetwork(state_dim).to(device)
        self.optimizer = optim.Adam([
            {'params': self.policy_net.parameters(), 'lr': lr},
            {'params': self.value_net.parameters(), 'lr': 3e-4}
        ])
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda ep: max(0.1, 1 - ep/5000)
        )
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.saved_log_probs = []
        self.rewards = []
        self.entropies = []
        self.states = []
        
    def select_action(self, state):
        state = torch.FloatTensor(state).to(device)
        probs = self.policy_net(state)
        dist = Categorical(probs)
        action = dist.sample()
        self.saved_log_probs.append(dist.log_prob(action))
        self.entropies.append(dist.entropy())
        return action.item()
    
    def update(self):
        # 計算折扣報酬
        returns = []
        R = 0
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # 狀態歸一化
        states = torch.FloatTensor(np.array(self.states)).to(device)
        states = (states - states.mean(0)) / (states.std(0) + 1e-8)  # 新增狀態歸一化
        
        # 計算優勢
        values = self.value_net(states).squeeze()
        advantages = returns - values.detach()
        
        # 損失計算
        policy_loss = -torch.stack(self.saved_log_probs) * advantages
        entropy_loss = -torch.stack(self.entropies).mean()
        policy_loss = policy_loss.mean() + self.entropy_coef * entropy_loss
        
        value_loss = F.smooth_l1_loss(values, returns)  # 改用平滑L1損失
        
        # 梯度裁剪與更新
        self.optimizer.zero_grad()
        (policy_loss + value_loss).backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        # 重置緩存
        self.saved_log_probs = []
        self.rewards = []
        self.entropies = []
        self.states = []
        return policy_loss.item(), value_loss.item()

def train(env, agent, max_episodes=5000):
    ewma_reward = 0
    for episode in range(max_episodes):
        state, _ = env.reset()
        total_reward = 0
        while True:
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.rewards.append(reward)
            agent.states.append(state)
            total_reward += reward
            state = next_state
            if done:
                break
        
        # 每回合更新一次
        policy_loss, value_loss = agent.update()
        
        # 更新EWMA獎勵
        ewma_reward = 0.05 * total_reward + 0.95 * ewma_reward
        
        # 記錄數據
        writer.add_scalar("Reward/Episode", total_reward, episode)
        writer.add_scalar("Reward/EWMA", ewma_reward, episode)
        writer.add_scalar("Loss/Policy", policy_loss, episode)
        writer.add_scalar("Loss/Value", value_loss, episode)
        
        if episode % 50 == 0:
            print(f"Episode {episode:4d} | Reward: {total_reward:7.1f} | EWMA: {ewma_reward:7.1f}")
            
        if ewma_reward >= 200:
            torch.save(agent.policy_net.state_dict(), "lunar_optimized.pth")
            print(f"Solved at Episode {episode}!")
            break

if __name__ == "__main__":
    env = gym.make("LunarLander-v2")
    agent = REINFORCE(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        lr=1e-4,
        gamma=0.995,
        entropy_coef=0.15
    )
    train(env, agent)
    env.close()