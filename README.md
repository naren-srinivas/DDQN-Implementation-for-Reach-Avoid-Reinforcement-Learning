# DDQN Implementation for Reach-Avoid Reinforcement Learning

This repository implements a Double Deep Q-Network (DDQN) approach for solving reach-avoid problems using the Discounted Reach-Avoid Bellman Equation (DRABE) as proposed in RSS21. The implementation focuses on 2D point mass navigation tasks where an agent must reach a target set while avoiding failure regions.

## Repository Structure

```
.
├── sim_naive.py                    # Main training script
├── RARL/                          # Core RL algorithms
│   ├── DDQNSingle.py              # DDQN implementation with DRABE
│   ├── DDQN.py                    # Base DDQN class
│   ├── model.py                   # Neural network architectures
│   ├── config.py                  # Configuration management
│   ├── utils.py                   # Utility functions
│   └── ReplayMemory.py           # Experience replay buffer
├── gym_reachability/              # Custom Gym environment
│   └── gym_reachability/
│       ├── __init__.py
│       └── envs/
│           ├── __init__.py
│           ├── zermelo_show.py    # 2D navigation environment
│           └── env_utils.py       # Environment utilities
└── experiments/                   # Training results and outputs
    └── naive/ddqn/RA/anneal-toEnd/
        ├── figures/               # Visualization outputs
        │   ├── env.png           # Environment visualization
        │   ├── initQ.png         # Initial Q-function
        │   ├── initQ_loss.png    # Warmup training loss
        │   ├── train_loss_success.png  # Training metrics
        │   ├── value_rollout_action.png # Final results
        │   ├── 300000.png        # Intermediate checkpoints
        │   └── 600000.png
        └── model/                # Saved model checkpoints (.pth files)
```

## Core Components

### 1. DDQNSingle Class (`RARL/DDQNSingle.py`)

The main DDQN implementation with reach-avoid modifications:

- **Key Methods**:
  - `update()`: Implements DRABE update rule
  - `initQ()`: Warmup initialization using heuristic values
  - `learn()`: Main training loop with periodic evaluation
  - `select_action()`: ε-greedy action selection

- **DRABE Implementation**:
  ```python
  # Non-terminal states
  non_terminal = torch.max(
      g_x[non_final_mask],
      torch.min(l_x[non_final_mask], state_value_nxt[non_final_mask]),
  )
  terminal = torch.max(l_x, g_x)
  
  y[non_final_mask] = non_terminal * self.GAMMA + terminal[non_final_mask] * (1 - self.GAMMA)
  ```

### 2. Zermelo Environment (`gym_reachability/envs/zermelo_show.py`)

2D point mass navigation environment featuring:
- Continuous state space (position)
- Discrete action space (velocity directions)
- Target and failure regions
- Margin functions ℓ(x) and g(x)
- Trajectory simulation and visualization

### 3. Training Script (`sim_naive.py`)

Main script for training and evaluation with support for:
- Multiple training modes (RA, Lagrange)
- Hyperparameter annealing
- Extensive visualization
- Model checkpointing

## Usage

### Basic Training Commands

**Reach-Avoid Training:**
```bash
# Standard RA training with annealing
python3 sim_naive.py -w -sf -of scratch -a -g 0.99 -n anneal

# RA training without annealing
python3 sim_naive.py -w -sf -of scratch -n 9999

# RA training with early termination on failure
python3 sim_naive.py -w -sf -of scratch -g 0.999 -dt fail -n 999
```


### Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `-w, --warmup` | Enable Q-network warmup | False |
| `-sf, --storeFigure` | Save visualization figures | False |
| `-a, --annealing` | Enable gamma annealing | False |
| `-g, --gamma` | Discount factor | 0.9999 |
| `-m, --mode` | Training mode (RA/lagrange) | 'RA' |
| `-dt, --doneType` | Episode termination condition | 'toEnd' |
| `-mu, --maxUpdates` | Maximum gradient updates | 400000 |
| `-cp, --checkPeriod` | Evaluation period | 20000 |

## Key Algorithmic Innovations

### 1. Heuristic Warmup

The implementation includes a warmup phase where the Q-network is initialized using heuristic values:

```python
def initQ(self, env, warmupIter, ...):
    states, heuristic_v = env.get_warmup_examples(num_warmup_samples)
    # Train Q-network to match heuristic values
    loss = mse_loss(input=v, target=heuristic_v)
```

### 2. Gamma Annealing

For improved convergence, the discount factor γ can be gradually increased during training:

```python
if args.annealing:
    GAMMA_END = 0.999999
    EPS_PERIOD = int(updatePeriod / 10)
```

## Experimental Results

The repository generates several types of visualizations:

1. **Environment Setup** (`env.png`): Shows ℓ(x), g(x), and v(x) = max{ℓ(x), g(x)}
2. **Training Progress** (`train_loss_success.png`): Loss curves and success rates
3. **Final Policy** (`value_rollout_action.png`): Learned value function, rollout results, and action distribution

### Performance Metrics

- **Success Rate**: Percentage of trajectories reaching the target
- **Failure Rate**: Percentage of trajectories entering failure set
- **Unfinished Rate**: Percentage of trajectories that timeout

## Installation and Dependencies

```bash
# Core dependencies
pip install torch torchvision numpy matplotlib gym

# Install custom environment
pip install -e gym_reachability/
```

## Theoretical Foundation

This implementation is based on the paper that introduces DRABE for reach-avoid problems. The key insight is that traditional discounted cost formulations don't directly address the reach-avoid objective, leading to suboptimal policies in safety-critical scenarios.
