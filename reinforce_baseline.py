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

# 定義儲存動作資訊的元組，包括對數概率和狀態值
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

# 定義 TensorBoard 記錄器
writer = SummaryWriter("./tb_record_baseline_optimized")
        
class Policy(nn.Module):
    """
    REINFORCE with Baseline 的策略網絡和值網絡實現
    - 保留部分特徵提取層共享，但後續層分離以減少干擾
    - 針對複雜的 LunarLander-v2 環境使用更深的網絡結構
    """
    def __init__(self, hidden_size=256):
        super(Policy, self).__init__()
        
        # 從環境中獲取狀態和動作的維度
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.observation_dim = env.observation_space.shape[0]  # LunarLander-v2 有 8 個維度
        self.action_dim = env.action_space.n  # LunarLander-v2 有 4 個維度
        self.hidden_size = hidden_size  # 增加隱藏層大小以提高網絡表達能力
        
        # 共享特徵提取層
        self.shared_fc = nn.Linear(self.observation_dim, self.hidden_size)
        
        # 策略網絡專用層
        self.policy_fc = nn.Linear(self.hidden_size, self.hidden_size)
        self.action_head = nn.Linear(self.hidden_size, self.action_dim)
        
        # 值網絡專用層 - 作為 baseline 函數
        self.value_fc = nn.Linear(self.hidden_size, self.hidden_size)
        self.value_head = nn.Linear(self.hidden_size, 1)

        # 權重初始化（使用 Xavier 初始化以改善深度網絡訓練）
        nn.init.xavier_uniform_(self.shared_fc.weight)
        nn.init.xavier_uniform_(self.policy_fc.weight)
        nn.init.xavier_uniform_(self.value_fc.weight)
        nn.init.xavier_uniform_(self.action_head.weight)
        nn.init.xavier_uniform_(self.value_head.weight)
        
        # 用於存儲動作和獎勵的記憶體
        self.saved_actions = []
        self.rewards = []
        self.entropies = []  # 儲存熵用於正則化

    def forward(self, state):
        """
        策略網絡和值網絡的前向傳播
        - 輸入是狀態，輸出是相應的動作概率分佈和狀態值
        """
        # 共享特徵提取層
        shared_features = F.relu(self.shared_fc(state))
        
        # 策略網絡路徑
        policy_features = F.relu(self.policy_fc(shared_features))
        action_logits = self.action_head(policy_features)
        action_prob = F.softmax(action_logits, dim=-1)
        
        # 值網絡路徑（baseline）
        value_features = F.relu(self.value_fc(shared_features))
        state_value = self.value_head(value_features)
        
        return action_prob, state_value

    def select_action(self, state):
        """
        根據當前狀態選擇動作
        - 輸入是狀態，輸出是要執行的動作（基於學習的隨機策略）
        """
        # 將狀態轉換為張量並增加批次維度
        state = torch.from_numpy(state).float().unsqueeze(0)
        
        # 獲取動作概率和狀態值
        action_prob, state_value = self.forward(state)
        
        # 創建類別分佈
        m = Categorical(action_prob)
        
        # 從分佈中採樣動作
        action = m.sample()
        
        # 計算熵用於正則化
        entropy = m.entropy()
        self.entropies.append(entropy)
        
        # 將動作的對數概率和狀態值保存到記憶體
        self.saved_actions.append(SavedAction(m.log_prob(action), state_value))
        
        return action.item()

    def calculate_loss(self, gamma=0.99, entropy_coef=0.005):
        """
        計算損失（策略損失 + 值損失 + 熵正則化）用於反向傳播
        - 策略損失使用基準線調整的策略梯度
        - 值損失使用均方誤差
        - 熵正則化鼓勵探索
        """
        R = 0
        saved_actions = self.saved_actions
        policy_losses = [] 
        value_losses = [] 
        returns = []

        # 從後向前計算累積折現回報
        for r in self.rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)
            
        returns = torch.tensor(returns, dtype=torch.float)
        
        # 標準化回報（提高訓練穩定性）
        if len(returns) > 1:  # 確保有多個回報以計算標準差
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        # 計算策略損失和值損失
        for i, ((log_prob, value), R) in enumerate(zip(saved_actions, returns)):
            # 使用狀態值作為基準線，計算優勢
            advantage = R - value.detach().squeeze(-1)
            
            # 策略損失：對數概率乘以優勢
            policy_losses.append(-log_prob * advantage)
            
            # 值損失：預測值與實際回報之間的差異
            value_losses.append(F.mse_loss(value.squeeze(-1), torch.tensor([R], dtype=torch.float)))
        
        # 計算熵正則化損失
        entropy_loss = -torch.stack(self.entropies).mean()
        
        # 結合損失，值損失權重保持在 0.5，增加熵正則化
        loss = torch.stack(policy_losses).sum() + 0.5 * torch.stack(value_losses).sum() + entropy_coef * entropy_loss
        
        return loss

    def clear_memory(self):
        """
        清除獎勵和動作記憶體
        """
        del self.rewards[:]
        del self.saved_actions[:]
        del self.entropies[:]


def train(lr=0.0001, gamma=0.99, entropy_coef=0.005, hidden_size=256, update_frequency=1):
    """
    使用 SGD（通過反向傳播）訓練模型
    - 執行策略直到回合結束，保存採樣的軌跡
    - 在回合結束時更新策略和值網絡
    - 在 TensorBoard 上記錄值以進行可視化
    """
    # 初始化策略模型和優化器
    model = Policy(hidden_size=hidden_size)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 添加學習率調度器以幫助收斂
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.9)
    
    # 用於追踪學習進度的 EWMA 獎勵
    ewma_reward = 0
    
    # 批次更新計數器
    batch_count = 0
    
    # 運行訓練回合
    for i_episode in count(1):
        # 重置環境和回合獎勵
        state, _ = env.reset()
        ep_reward = 0
        t = 0
        
        # 運行完整回合
        while True:
            # 選擇動作
            action = model.select_action(state)
            
            # 執行動作
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated            
            
            # 保存獎勵
            model.rewards.append(reward)
            ep_reward += reward
            t += 1
            
            # 回合結束處理
            if done:
                break
            
        # 更新 EWMA 獎勵和記錄結果
        ewma_reward = 0.05 * ep_reward + (1 - 0.05) * ewma_reward
        batch_count += 1

        # 記錄 TensorBoard 值
        writer.add_scalar('Reward/Episode', ep_reward, i_episode)
        writer.add_scalar('Episode_Length', t, i_episode)
        writer.add_scalar('EWMA_Reward', ewma_reward, i_episode)
        writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], i_episode)

        # 每隔 update_frequency 回合更新一次網絡，或者 EWMA 獎勵足夠高時更新
        if batch_count >= update_frequency or ewma_reward > 200:
            # 計算損失並更新網絡
            loss = model.calculate_loss(gamma, entropy_coef)
            writer.add_scalar('Loss/Total', loss.item(), i_episode)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            model.clear_memory()
            batch_count = 0
        
        # 每 100 個回合打印一次回合信息
        if i_episode % 100 == 0:
            print('Episode {}\tLength: {}\tReward: {:.2f}\tEWMA Reward: {:.2f}'.format(
                i_episode, t, ep_reward, ewma_reward))
        
        # 保存模型並完成訓練，LunarLander-v2 的解決閾值是 200
        if ewma_reward > 200:
            if not os.path.isdir("./preTrained"):
                os.mkdir("./preTrained")
            torch.save(model.state_dict(), './preTrained/LunarLander_baseline_optimized.pth')
            print("已解決！運行獎勵為 {:.2f}，最後回合運行了 {} 個時間步！".format(ewma_reward, t))
            break


def test(name, n_episodes=10, hidden_size=256):
    """
    測試學習的模型
    """     
    model = Policy(hidden_size=hidden_size)
    
    model.load_state_dict(torch.load('./preTrained/{}'.format(name)))
    
    render = True
    max_episode_len = 1000
    
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
    # 設置隨機種子以確保可重複性
    random_seed = 10  
    
    # 超參數設置
    lr = 0.0001          # 更低的學習率以穩定訓練
    gamma = 0.99         # 稍低的折扣因子
    entropy_coef = 0.005 # 新增的熵正則化係數
    hidden_size = 256    # 更大的隱藏層大小提高表達能力
    update_frequency = 1 # 每回合更新，可以設置更高的值以加速訓練
    
    # 創建 LunarLander-v2 環境
    env = gym.make('LunarLander-v2', render_mode="rgb_array")
    torch.manual_seed(random_seed)  
    
    # 訓練模型
    train(lr, gamma, entropy_coef, hidden_size, update_frequency)
    
    # 測試訓練好的模型
    test('LunarLander_baseline_optimized.pth', hidden_size=hidden_size)