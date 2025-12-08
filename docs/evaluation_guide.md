# Evaluation Guide

This guide explains how to evaluate trained RL agents against baselines and against each other.

## Overview

The evaluation system supports three types of testing:
1. **Baselines Only** - Compare rule-based baselines (Random, EV Oracle, Level-1)
2. **RL vs Baselines** - Test trained agents against rule-based baselines
3. **RL vs RL** - Test trained agents against each other (self-play and cross-play)

## 1. Evaluating Baselines Only

Evaluate the performance of rule-based baseline agents:

```bash
python scripts/eval_baselines.py --n-episodes 100 --output data/results/baselines.csv
```

This tests:
- **Random Valid**: Random action selection from valid actions
- **EV Oracle** (privileged): Perfect information baseline (knows true fair value from environment)
- **EV Realistic** (no privileged info): Bayesian baseline using only observable data
- **Level-1** (privileged): Strategic baseline with opponent modeling (uses `info["mu"]`)
- **Level-1 Realistic** (no privileged info): Strategic baseline with opponent modeling, computing fair value from observable data only

### Baseline Performance Hierarchy (Expected)

**Privileged Baselines** (theoretical upper bounds):
1. **EV Oracle** (strongest) - Perfect information
2. **Level-1** - Strategic with perfect fair value

**Realistic Baselines** (achievable without cheating):
3. **Level-1 Realistic** - Strategic with Bayesian fair value
4. **EV Realistic** - Bayesian without opponent modeling
5. **Trained RL Agents** - Should approach Level-1 Realistic
6. **Random Valid** (weakest) - Baseline floor

### Why Two Hierarchies?

The gap between privileged and realistic baselines represents the **information asymmetry cost**:
- EV Oracle vs EV Realistic: Cost of not knowing true μ
- Level-1 vs Level-1 Realistic: Cost of computing vs knowing μ

RL agents should aim to match realistic baselines, not privileged ones.

## 2. Evaluating RL Agents vs Baselines

### Option A: Using `compare_agents.py` (Comprehensive)

Test a single agent against all baselines:

```bash
# Test IPPO against baselines
python -m mmrl.eval.compare_agents \
    --agent ippo \
    --model_path data/models/ippo/ippo_final.pt \
    --episodes 100 \
    --output data/results/ippo_vs_baselines.csv

# Test MAPPO against baselines
python -m mmrl.eval.compare_agents \
    --agent mappo \
    --model_path data/models/mappo/mappo_final.pt \
    --episodes 100 \
    --output data/results/mappo_vs_baselines.csv

# Test DQN against baselines (single-player comparison)
python -m mmrl.eval.compare_agents \
    --agent dqn \
    --model_path data/models/dqn/dqn_final.pt \
    --episodes 100 \
    --output data/results/dqn_comparison.csv
```

### Option B: Using `eval_all.py` (Simplified)

Quick evaluation of all agents:

```bash
python scripts/eval_all.py --n-episodes 100 --output data/results/eval_all.csv
```

Note: This currently tests against Random opponent only (simpler but less comprehensive).

## 3. Evaluating RL Agents Against Each Other

### Option A: Test All Agent Combinations

Test all trained agents against each other (including self-play):

```bash
# Test both IPPO and MAPPO (all combinations)
python scripts/eval_rl_vs_rl.py \
    --agents ippo mappo \
    --episodes 100 \
    --output data/results/rl_vs_rl.csv

# Self-play only (MAPPO vs MAPPO)
python scripts/eval_rl_vs_rl.py \
    --agents mappo \
    --episodes 100 \
    --output data/results/mappo_self_play.csv

# Skip self-play matchups
python scripts/eval_rl_vs_rl.py \
    --agents ippo mappo \
    --no-self-play \
    --episodes 100 \
    --output data/results/rl_cross_play.csv
```

### Option B: Test Specific Matchup

Test a specific agent pair:

```bash
# IPPO vs MAPPO
python -m mmrl.eval.compare_agents \
    --agent ippo \
    --opponent mappo \
    --episodes 100 \
    --output data/results/ippo_vs_mappo.csv

# MAPPO vs IPPO (reverse matchup)
python -m mmrl.eval.compare_agents \
    --agent mappo \
    --opponent ippo \
    --episodes 100 \
    --output data/results/mappo_vs_ippo.csv

# Self-play: IPPO vs IPPO
python -m mmrl.eval.compare_agents \
    --agent ippo \
    --opponent ippo \
    --episodes 100 \
    --output data/results/ippo_self_play.csv
```

### Using Custom Checkpoints

Test different training checkpoints:

```bash
# Compare early vs late IPPO checkpoint
python -m mmrl.eval.compare_agents \
    --agent ippo \
    --model_path data/models/ippo/ippo_step10000.pt \
    --opponent ippo \
    --opponent_path data/models/ippo/ippo_final.pt \
    --episodes 100
```

## Understanding the Output

### Baseline Evaluation Output
```
agent          return_mean  return_std  sharpe    valid_rate  n_episodes
Random         -2.5         15.3        -0.163    1.0         100
EV_Oracle      12.8         8.2         1.561     1.0         100
Level1         10.5         9.1         1.154     1.0         100
```

### RL vs Baselines Output
```
opponent       mean_return_rl  std_return_rl  mean_return_opp  std_return_opp
random_valid   8.5             10.2           -3.2             12.1
ev_oracle      -5.2            15.8           18.3             9.5
level1         3.1             12.5           7.8              11.2
```

### RL vs RL Output
```
agent_a  agent_b  mean_return_a  std_return_a  sharpe_a  mean_return_b  std_return_b  sharpe_b
IPPO     IPPO     2.3            11.5          0.200     2.1            11.8          0.178
IPPO     MAPPO    4.2            10.8          0.389     3.8            11.2          0.339
MAPPO    IPPO     3.5            11.1          0.315     4.5            10.5          0.429
MAPPO    MAPPO    1.8            12.0          0.150     1.9            11.8          0.161
```

## Metrics Explained

- **return_mean**: Average cumulative reward per episode
- **return_std**: Standard deviation of returns (risk measure)
- **sharpe**: Risk-adjusted return (mean/std) - higher is better
- **valid_rate**: Fraction of valid actions (should be 1.0)
- **ruin_prob**: Probability of hitting stop-out threshold
- **mean_return_a/b**: Returns for player A/B in two-player games

## Performance Interpretation

### Expected Performance Hierarchy

1. **EV Oracle** (strongest) - Has perfect information
2. **Level-1 Policy** (strong) - Strategic with game theory
3. **Trained RL Agents** (variable) - Should approach Level-1 performance
4. **Random Valid** (weakest) - Baseline floor

### Good RL Agent Indicators

- Positive return against Random opponent
- Sharpe ratio > 0.3
- Return within 30-50% of EV Oracle
- Better than Random in >95% of episodes

### Self-Play Insights

- **Balanced self-play** (returns ≈0 for both): Agent has converged
- **Asymmetric self-play**: Positional advantage or stochasticity
- **High variance**: Agent hasn't fully converged or environment is high-variance

## Advanced Usage

### Custom Environment Config

Create a config file `custom_eval.yaml`:

```yaml
episode_length: 20
W0: 500.0
flags:
  enable_events: true
  enable_impact: false
stop_out: 0.1
pass_penalty: 0.05
```

Then modify the evaluation scripts to load this config.

### Batch Evaluation

Run multiple evaluations:

```bash
#!/bin/bash
# Evaluate all agents against all opponents

for agent in ippo mappo; do
    echo "Evaluating $agent..."
    
    # vs baselines
    python -m mmrl.eval.compare_agents \
        --agent $agent \
        --episodes 100 \
        --output data/results/${agent}_vs_baselines.csv
    
    # self-play
    python -m mmrl.eval.compare_agents \
        --agent $agent \
        --opponent $agent \
        --episodes 100 \
        --output data/results/${agent}_self_play.csv
done

# Cross-play
python scripts/eval_rl_vs_rl.py \
    --agents ippo mappo \
    --no-self-play \
    --episodes 100 \
    --output data/results/cross_play.csv
```

## Troubleshooting

### Model Not Found
```
Error: IPPO checkpoint not found at data/models/ippo/ippo_final.pt
```
**Solution**: Train the model first or specify correct path with `--model_path`

### Import Errors
```
ModuleNotFoundError: No module named 'mmrl'
```
**Solution**: Run from project root or install package with `pip install -e .`

### CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solution**: Models automatically use CPU if CUDA unavailable. Force CPU with `device="cpu"` in code.

### Different Training/Eval Configs
If evaluation performance seems poor, ensure eval config matches training config:
- Same `episode_length`
- Same `enable_events`, `enable_impact` flags
- Same `pass_penalty`

## Next Steps

After evaluation:
1. Compare agent performance in `data/results/`
2. Visualize results using `diagnostics.py`
3. Run out-of-distribution tests using `ood_tests.py`
4. Iterate on training hyperparameters based on results

