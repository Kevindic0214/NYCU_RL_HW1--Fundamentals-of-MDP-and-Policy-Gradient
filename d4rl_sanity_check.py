#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""檢查 Gymnasium 和 Gymnasium-Robotics 中的可用環境"""

import gymnasium as gym
import gymnasium_robotics  # 顯式導入以確保其環境被註冊

# 列出所有已註冊的環境
from gymnasium.envs.registration import registry
all_envs = list(registry.keys())

# 查找任何可能相關的環境
maze_related = [env for env in all_envs if any(keyword in env.lower() for keyword in ['maze', 'd4rl', 'offline'])]
print("可能與迷宮或離線 RL 相關的環境：")
for env in maze_related:
    print(f"  - {env}")

# 檢查 Gymnasium-Robotics 是否正確導入
print("\nGymnasium-Robotics 版本和模塊信息：")
try:
    print(f"版本: {gymnasium_robotics.__version__}")
    print(f"可用子模塊: {dir(gymnasium_robotics)}")
except Exception as e:
    print(f"無法獲取版本信息: {e}")

# 嘗試導入特定的子模塊
print("\n嘗試直接從 gymnasium_robotics 導入可能的迷宮模塊：")
try:
    # 嘗試各種可能的子模塊名稱
    potential_modules = ['maze', 'mazes', 'd4rl', 'offline']
    for module in potential_modules:
        try:
            # 使用 __import__ 動態導入
            imported_module = __import__(f'gymnasium_robotics.{module}', fromlist=[''])
            print(f"成功導入 gymnasium_robotics.{module}")
            print(f"子模塊內容: {dir(imported_module)}")
        except ImportError as e:
            print(f"無法導入 gymnasium_robotics.{module}: {e}")
except Exception as e:
    print(f"探索子模塊時發生錯誤: {e}")

# 探索 Minari 數據集
print("\n探索 Minari 中的可用數據集：")
try:
    import minari
    datasets = minari.list_datasets()
    print(f"可用數據集: {datasets}")
    maze_datasets = [ds for ds in datasets if 'maze' in ds.lower()]
    print(f"迷宮相關數據集: {maze_datasets}")
except Exception as e:
    print(f"無法獲取 Minari 數據集信息: {e}")