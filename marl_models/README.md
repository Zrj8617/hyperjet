# MARL Algorithms - Multi-Agent Reinforcement Learning Models

This folder contains 8 different MARL algorithm implementations for multi-agent coordination along with a random baseline which selects random actions.

## 📁 Folder Structure

```
marl_models/
├── base_model.py          # Base class for all models
├── buffer_and_helpers.py  # Experience replay buffer
├── utils.py               # Model factory & utilities
├── README.md              # README for the folder
│
├── maddpg/                # MADDPG implementation
│   ├── agents.py          # Agent class
│   └── maddpg.py          # Algorithm update logic
│
├── matd3/                 # MATD3 implementation
├── mappo/                 # MAPPO implementation
├── masac/                 # MASAC implementation
│
├── attention.py           # Attention base classes
|
├── attention_maddpg/      # MADDPG + Attention
│   ├── agents.py
│   └── attention_maddpg.py
│
├── attention_matd3/       # MATD3 + Attention
├── attention_mappo/       # MAPPO + Attention
├── attention_masac/       # MASAC + Attention
│
└── random_baseline/       # Random baseline
    └── random_model.py
```

## 🔧 How to Use

### Switch Between Algorithms

Simply change `MODEL` in [config.py](../config.py):

```python
# Choose one:
MODEL = "maddpg"              # Off-policy baseline
MODEL = "matd3"               # Twin delayed DDPG
MODEL = "mappo"               # On-policy PPO
MODEL = "masac"               # Soft Actor-Critic
MODEL = "attention_maddpg"    # MADDPG + attention
MODEL = "attention_matd3"     # MATD3 + attention
MODEL = "attention_mappo"     # MAPPO + attention
MODEL = "attention_masac"     # MASAC + attention
MODEL = "random"              # Random baseline
```

## 🎯 Tuning Hyperparameters

### Stage 1: Reward Optimization
No algorithm-specific tuning, just reward weights:
```bash
python tune.py --stage 1 --episodes 500 --trials 50
```

### Stage 2: Algorithm Hyperparameters
Tunes learning rates, network size, batch size, discount factor etc:
```bash
python tune.py --stage 2 --episodes 1000 --trials 50
```

### Stage 3: Attention Architecture (Attention Models Only)
Optimize attention dimension and heads:
```bash
python tune.py --stage 3 --episodes 500 --trials 30
```

## 🔍 Comparing Algorithms

```bash
# Train using multiple algorithms by changing model in configs.py

# Compare results
python utils/comparative_plots.py \
  --logs train_logs/maddpg train_logs/masac/ \
  --names MADDPG MASAC \
  --smoothing 10
```

Generates comparison plots showing:
- Reward curves (learning progress)
- Latency (task completion time)
- Energy consumption
- Fairness (equal service)
- Offline rate (battery health)
- Loss curves (training stability)

Refer [Plotting Module](/docs/PLOTTING_MODULE.md) for detailed plotting plan.
