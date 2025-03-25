# RL Homework 1 - Fundamentals of MDPs and Policy Gradient

📘 **Course**: 535514 Reinforcement Learning (Spring 2025)  
📅 **Due Date**: March 26, 2025  
👨‍🎓 **Student**: [Kevin H. Hsieh / 110704054]  

## 📄 Overview

This repository contains my solutions for **Homework 1** of the Reinforcement Learning course. The homework is designed to test understanding of fundamental RL concepts such as:

- Markov Decision Processes (MDPs)
- Bellman Optimality Equations
- Q-Value Iteration
- Policy Gradient Methods (with Baselines and GAE)
- Function Approximation with PyTorch
- Offline RL using MuJoCo and D4RL

---

## 📁 Repository Structure

```bash
.
├── report.pdf                  # Technical write-up
├── code/
│   ├── reinforce.py            # Vanilla REINFORCE on CartPole-v0
│   ├── reinforce_baseline.py  # REINFORCE + Baseline on LunarLander-v2
│   ├── reinforce_gae.py       # REINFORCE + GAE on LunarLander-v2
│   └── d4rl_sanity_check.py   # Code for D4RL environment
├── models/
│   ├── cartpole_model.pth     # Trained model for CartPole
│   ├── lunarlander_model.pth  # Trained model for LunarLander
│   └── gae_models/            # Folder with models for different lambda values
└── README.md                  # Project documentation
