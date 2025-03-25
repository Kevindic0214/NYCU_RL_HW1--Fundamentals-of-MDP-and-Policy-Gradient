# RL Homework 1 - Fundamentals of MDPs and Policy Gradient

📘 **Course**: 535514 Reinforcement Learning (Spring 2025)  
🗓 **Due Date**: March 26, 2025  
👨‍🎓 **Student**: [Your Name / 學號]

## 📄 Overview

This repository contains my solutions for **Homework 1** of the Reinforcement Learning course. The homework explores theoretical foundations and practical implementations of reinforcement learning algorithms, including:

* Markov Decision Processes (MDPs)
* Bellman Optimality Equations
* Q-Value Iteration
* Policy Gradient Methods (with Baselines and GAE)
* Function Approximation with PyTorch
* Offline RL using MuJoCo and D4RL

## 📁 Repository Structure

```bash
.
├── report.pdf                  # Technical write-up
├── code/
│   ├── reinforce.py            # Vanilla REINFORCE on CartPole-v0
│   ├── reinforce_baseline.py   # REINFORCE + Baseline on LunarLander-v2
│   ├── reinforce_gae.py        # REINFORCE + GAE on LunarLander-v2
│   └── d4rl_sanity_check.py    # Code for D4RL environment
├── models/
│   ├── cartpole_model.pth      # Trained model for CartPole
│   ├── lunarlander_model.pth   # Trained model for LunarLander
│   └── gae_models/             # Folder with models for different lambda values
└── README.md                   # Project documentation
```

## 🧠 Problem Breakdown

### 🟩 Problem 1: Bellman Optimality & Q-Value Iteration
* Prove the Bellman equations.
* Implement Q-value iteration.
* Analyze γ-contraction properties under both ℓ∞ and ℓ1 norms.

### 🟩 Problem 2: Policy Gradient Property
* Mathematically prove a useful expectation identity used in policy gradient methods.

### 🟩 Problem 3: Baseline for Variance Reduction
* Calculate expected gradient and covariance matrix.
* Demonstrate variance reduction using value function baseline.
* Identify optimal baseline that minimizes trace of covariance.

### 🟩 Problem 4: Policy Gradient Algorithms
* **Vanilla REINFORCE** on `CartPole-v0`
  - Simple neural network policy
  - Hyperparameters to be determined through experimentation
  
* **REINFORCE with Baseline** on `LunarLander-v2`
  - State-value function approximator
  - Separate networks for policy and value functions
  
* **REINFORCE with GAE** on `LunarLander-v2`
  - Experimenting with different λ values (0.8, 0.9, 0.95)
  - Performance comparison between configurations

* Training progress to be logged using **TensorBoard**

### 🟩 Problem 5: D4RL and MuJoCo
* Install `d4rl` and `gymnasium`.
* Generate offline datasets and analyze their structure.

## 🛠️ Setup

To run the code:

```bash
conda create -n rl_hw1 python=3.10
conda activate rl_hw1
pip install -r requirements.txt
```

### Recommended Packages
* `torch`
* `gymnasium`
* `tensorboard`
* `wandb`
* `d4rl`
* `mujoco`
* `numpy`, `matplotlib`

## 📊 Implementation Progress

This is a work in progress. The following tasks are planned:

- [ ] Theoretical proofs for Problems 1-3
- [ ] Implementation of vanilla REINFORCE for CartPole-v0
- [ ] Implementation of REINFORCE with baseline for LunarLander-v2
- [ ] Implementation of REINFORCE with GAE for LunarLander-v2
- [ ] D4RL environment setup and exploration
- [ ] Technical report summarizing findings and results

## 📈 TensorBoard Samples

Upon completion, this section will include screenshots of tensorboard logs showing learning curves for:
* Average Return (CartPole) 
* LunarLander performance with REINFORCE + Baseline
* Comparative performance with different GAE λ values

## 📚 References

* [PyTorch Tutorials](https://pytorch.org/tutorials/)
* [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) (Schulman et al., 2015)
* [D4RL: Datasets for Deep Data-Driven Reinforcement Learning](https://arxiv.org/abs/2004.07219)
* [D4RL GitHub Repository](https://github.com/Farama-Foundation/D4RL)

## 📬 To Be Submitted

- [ ] `report.pdf` 
- [ ] All source code in `code/` 
- [ ] Trained model files in `models/` 
- [ ] All experiments documented with scripts and results
