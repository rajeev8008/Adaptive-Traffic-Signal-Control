# Adaptive Traffic Signal Control using PPO Reinforcement Learning

## Project Overview

This project implements an intelligent traffic signal controller using **Proximal Policy Optimization (PPO)**, a state-of-the-art deep reinforcement learning algorithm. The system intelligently manages traffic lights in SUMO (Simulation of Urban Mobility) to prioritize emergency vehicles while maintaining optimal flow for all vehicle types.

### Key Achievement
**10.6% improvement** in emergency vehicle response time compared to fixed-time baseline control

---

## Supported Vehicle Types

The model intelligently handles six distinct vehicle types with different priorities:

| Vehicle Type | Priority | Weight | Description |
|---|---|---|---|
| Emergency | 1st | 5.0x | Ambulances, Fire Trucks |
| Truck | 2nd | 4.0x | Delivery, Commercial |
| Car | 3rd | 3.0x | Regular commuters |
| Auto/Taxi | 4th | 2.0x | Ride-sharing services |
| Motorcycle | 5th | 1.0x | Two-wheelers |
| Bus | 6th | 0.5x | Public transit |

---

## Performance Results

### Overall Comparison: Baseline vs PPO Agent (Trinity Network)

| Vehicle Type | Baseline | PPO Agent | Improvement |
|---|---|---|---|
| Emergency | 69.00s | 61.67s | **10.6%** |
| Truck | 176.05s | 168.57s | **4.2%** |
| Car | 153.45s | 148.30s | **3.4%** |
| Motorcycle | 148.48s | 146.65s | **1.2%** |

### Cross-Map Performance (Indiranagar Network)

| Vehicle Type | Baseline | PPO Agent | Improvement |
|---|---|---|---|
| Emergency | 25.73s | 23.83s | **7.4%** |
| Truck | 151.30s | 92.39s | **39.0%** |
| Bus | 90.96s | 87.49s | **3.8%** |
| Car | 82.28s | 84.39s | -2.6% |

---

## Project Structure

```
sumo-traffic-rl-project/
├── README.md                            # This file - start here!
├── requirements.txt                     # Python dependencies
│
├── SCRIPTS
├── scripts/
│   ├── train_ppo.py                     # Training script (150k steps)
│   ├── evaluate_ppo.py                  # Comprehensive evaluation
│   ├── baseline.py                      # Fixed-time baseline
│   └── visualize_model.py               # Agent behavior visualization
│
├── UTILITIES & CORE MODULES
├── utils/
│   ├── SumoEnv.py                       # SUMO environment wrapper
│   └── ppo_agent.py                     # PPO agent configuration
│
├── ANALYSIS & EXPLAINABILITY (XAI)
├── analysis/
│   ├── explain_reward.py                # Reward decomposition (vehicle priority)
│   ├── explain_shap.py                  # SHAP feature attribution analysis
│   └── reward_analysis.json             # Analysis results
│
├── DATA AND MODELS
├── models/ppo_mg_road/
│   ├── best_model.zip                   # Best trained model
│   └── final_model.zip                  # Final checkpoint
├── logs/
│   ├── ppo_training/
│   │   ├── config.json                  # Training configuration
│   │   └── validation_results.json      # Training metrics
│   ├── ppo_evaluation/                  # Evaluation results
│   └── policy_explanation/              # SHAP visualizations & analysis
│
└── TRAFFIC NETWORKS
    ├── SUMO_Trinity_Traffic_sim/        # Primary training network
    │   ├── osm.sumocfg                  # Trinity simulation config
    │   └── *.rou.xml                    # Route definitions
    └── SUMO_Indiranagar_Traffic_sim/    # Cross-map evaluation network
        ├── osm.sumocfg                  # Indiranagar simulation config
        └── *.rou.xml                    # Route definitions
```

---

## Quick Start (5 Minutes)

### Prerequisites
- Python 3.10+
- SUMO 1.19.0+ (with `SUMO_HOME` set)

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Verify setup
sumo --version
python -c "from SumoEnv import SumoEnv; print('Ready')"
```

### Usage
```bash
# Train a new model (~30-40 minutes, 150k steps)
python scripts/train_ppo.py --no-gui

# Evaluate trained model (fixes vs baseline)
python scripts/evaluate_ppo.py --model models/ppo_mg_road/best_model.zip

# Analyze reward decomposition (vehicle priority)
python analysis/explain_reward.py

# SHAP feature attribution (decision explainability)
python analysis/explain_shap.py
# Outputs: shap_summary_plot.png, shap_force_plot.html

# Monitor training with TensorBoard
tensorboard --logdir logs/ppo_training

# Visualize agent behavior
python scripts/visualize_model.py
```

---

## Training Configuration

Edit `train_ppo.py` to customize:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TOTAL_TIMESTEPS` | 150,000 | Total training steps |
| `EVAL_FREQ` | 10,000 | Evaluate every N steps |
| `N_EVAL_EPISODES` | 5 | Episodes per evaluation |
| `EARLY_STOPPING_PATIENCE` | 5 | Stop if no improvement for N evals |
| `N_ENVS` | 4 | Parallel environments |

---

## Core Components

### **SumoEnv.py** - SUMO Environment
- Observation space: 43 features (queue lengths, phase, vehicle counts)
- Action space: 2 actions (keep/switch traffic signal)
- Reward: Based on vehicle travel times with priority weights

### **ppo_agent.py** - PPO Implementation
- Anti-overfitting measures:
  - L2 regularization (weight decay)
  - Entropy coefficient = 0.02
  - Gradient clipping = 0.5
  - Orthogonal initialization
- Network: [256, 256, 128] hidden layers

### **train_ppo.py** - Training Script
- Multi-core training via SubprocVecEnv (4 parallel environments)
- Multi-seed validation during training
- Early stopping mechanism
- Automatic model checkpointing

### **evaluate_ppo.py** - Evaluation
- Tests on unseen random seeds
- Compares against baseline (fixed-time control)
- Reports improvement percentages
- Saves results to JSON

### **baseline.py** - Baseline Comparison
- Implements traditional fixed-time traffic signal control
- Uses same vehicle priority weights as PPO
- Provides performance baseline

---

## Explainable AI (XAI) Components

### **explain_reward.py** - Reward Decomposition Analysis
- Analyzes multi-objective reward contributions from the PPO agent
- Visualizes percentage contribution of each reward component:
  - Emergency vehicle wait time reduction (30%)
  - Traffic flow improvement (45%)
  - Truck wait time reduction (15%)
  - Car wait time reduction (10%)
- Proves the agent actively optimizes for emergency vehicles
- Generates JSON analysis file with detailed metrics

### **explain_shap.py** - SHAP Feature Attribution Analysis
- Uses SHAP (SHapley Additive exPlanations) for decision explainability
- Generates two visualizations:
  - **Summary Plot**: Global feature importance (bar chart of top 20 features)
  - **Force Plot**: Local explanation for specific emergency observation
- Shows which features drive the agent toward switching traffic phases
- Identifies queue lengths and emergency counts as primary decision factors
- Fully interpretable feature names (Queue_Lane_0, Emerg_Lane_1, etc.)

### Key XAI Features
- **Multi-layer Transparency**: Reward decomposition + feature attribution
- **Global & Local Explanations**: Summary plots + force plots for specific events
- **Vehicle Priority Validation**: Verify emergency vehicle optimization
- **Decision Interpretability**: Understand which traffic features influence agents

---

## How It Works

### Observation Space (43D)
- Queue lengths per lane (14D)
- Current signal phase (1D)
- Emergency vehicle counts (14D)
- Bus counts (14D)

### Action Space (2 Actions)
- **Action 0**: Keep current phase
- **Action 1**: Switch to next phase

### Reward Function
```
Total Reward =
  45% × (traffic flow reduction) +
  30% × (emergency wait time reduction) +
  15% × (truck wait time reduction) +
  10% × (car wait time reduction)
```

---

## Troubleshooting

### SUMO not found
```bash
# Windows PowerShell
$env:SUMO_HOME = "C:\Program Files\SUMO"

# Or permanent:
[System.Environment]::SetEnvironmentVariable("SUMO_HOME", "C:\Program Files\SUMO", "User")
```

### Low memory / Training slow
```bash
# Use headless mode (faster)
python train_ppo.py --no-gui

# Edit train_ppo.py for reduced size:
# TrainingConfig.TOTAL_TIMESTEPS = 100_000
# TrainingConfig.N_STEPS = 1024
```

### Port already in use
Wait 30 seconds and restart, or kill SUMO processes.

### Model not found
```bash
# Train first
python train_ppo.py --no-gui
# Wait ~30-40 minutes for completion
```

---

## Anti-Overfitting Safeguards

The implementation includes multiple safeguards against overfitting:

1. **L2 Regularization** - Weight decay in optimizer
2. **Entropy Coefficient** (0.02) - Encourages exploration
3. **Gradient Clipping** (0.5) - Prevents exploding gradients
4. **Multi-seed Validation** - Tests on different random seeds
5. **Early Stopping** - Halts if validation plateaus
6. **Conservative Architecture** - 3-layer network with orthogonal init
7. **Deterministic Evaluation** - Tests both stochastic & deterministic policies

---

## Usage Examples

### Example 1: Evaluate Model Performance (2 min)
```bash
python scripts/evaluate_ppo.py --model models/ppo_mg_road/best_model.zip
```
Produces: Performance comparison tables, improvement percentages

### Example 2: Complete Training Pipeline (30-40 min)
```bash
python scripts/train_ppo.py --no-gui
```
Produces: Trained model, TensorBoard logs, validation metrics

### Example 3: Full Analysis Suite (10 min)
```bash
python analysis/explain_reward.py    # Vehicle priority analysis
python analysis/explain_shap.py      # Feature importance analysis
```
Produces: JSON metrics, PNG plots, HTML force plots

### Example 4: Baseline Comparison
```bash
python scripts/baseline.py           # Fixed-time signal control
python scripts/evaluate_ppo.py       # Intelligent agent control
compare results                       # Review improvements
```

---

## Key Evaluation Metrics

During training, monitor in TensorBoard:
- **Ep Rew Mean** - Should increase (become less negative) over time
- **Validation metrics** - Watch for plateauing (triggers early stopping)

After evaluation:
- **Overfitting Status** - Should be "NO"
- **Performance Drop** - Should be < 10%
- **Improvement Percentage** - Emergency vehicles should show ~10% improvement

---

## Technical Details

| Property | Value |
|---|---|
| **Algorithm** | Proximal Policy Optimization (PPO) |
| **Framework** | Stable-Baselines3 + Gymnasium |
| **Simulation** | SUMO (Simulation of Urban Mobility) |
| **Network** | 2 hidden layers, 256 neurons each |
| **Training Steps** | 150,000 |
| **Python** | 3.10+ |
| **SUMO Version** | 1.19.0+ |

---

## Key Achievements

* **10.6% improvement** in emergency vehicle response time (Trinity)
* **39% improvement** in truck performance (Indiranagar, cross-map)
* **7.4% improvement** in emergency response (Indiranagar, cross-map)
* Intelligent prioritization of six vehicle types
* Robust generalization across different traffic networks
* Production-ready implementation with comprehensive logging

---

**Status**: Production Ready | **Version**: 1.0 | **Updated**: February 2026
