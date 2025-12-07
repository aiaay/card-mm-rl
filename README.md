# Card-Sum Market Making RL

A reproducible research codebase for studying market-taking strategies in a stylized card-sum market with realistic microstructure (Tier-2 liquidity, impact, events).

## Overview
The goal is to trade (long/short) on the sum of 3 hidden cards (2-10, J, Q, K, A).
- **Information**: Hints (revealed cards), Market Quotes (Mid/Spread), Displayed Depth.
- **Dynamics**: Tier-2 liquidity (hidden vs displayed), market impact, slippage.
- **Events**: Probabilistic constraints (e.g., "Even cards only", "Sum >= 10").

See [docs/architecture.md](docs/architecture.md) for detailed system design.

## Installation

```bash
pip install -e .
```

Requirements: Python 3.10+, `torch`, `gymnasium`, `numpy`, `pandas`, `pyyaml`, `tensorboard`.

---

## Complete Training & Evaluation Workflow

### 1. Train All RL Agents

```bash
# Train DQN (Single Player) - 1000 episodes
python -m mmrl.agents.dqn.train_dqn --episodes 1000 --seed 42

# Train IPPO (Two Player) - 50K steps
python -m mmrl.agents.ippo.train_ippo --steps 50000 --seed 42

# Train MAPPO (Two Player) - 50K steps
python -m mmrl.agents.mappo.train_mappo --steps 50000 --seed 42

# Or train all at once with the bash script:
bash scripts/train_all.sh
```

**Outputs:**
- **Models**: Saved to `data/models/{dqn,ippo,mappo}/`
- **Logs**: TensorBoard logs in `data/logs/{dqn,ippo,mappo}/`

### 2. View Training Progress

```bash
tensorboard --logdir data/logs
```
Open `http://localhost:6006` to see training curves.

### 3. Evaluate All Agents & Baselines

**Evaluate Baselines** (Random, EV Oracle, Level-1):
```bash
python scripts/eval_baselines.py --n-episodes 100
```
Results saved to `data/results/baselines.csv`.

**Evaluate RL Agents vs Baselines**:
```bash
# Test IPPO against all baselines
python -m mmrl.eval.compare_agents --agent ippo --episodes 100

# Test MAPPO against all baselines  
python -m mmrl.eval.compare_agents --agent mappo --episodes 100

# Or use the simplified script (tests against random only)
python scripts/eval_all.py --n-episodes 100 --output data/results/eval_all.csv
```

**Evaluate RL Agents Against Each Other** (NEW!):
```bash
# Test all trained agents against each other (self-play + cross-play)
python scripts/eval_rl_vs_rl.py --agents ippo mappo --episodes 100

# Test specific matchup: IPPO vs MAPPO
python -m mmrl.eval.compare_agents --agent ippo --opponent mappo --episodes 100

# Self-play: IPPO vs IPPO
python -m mmrl.eval.compare_agents --agent ippo --opponent ippo --episodes 100

# Run all evaluations at once
bash scripts/run_all_evaluations.sh
```

This runs test episodes and saves metrics (Return, Sharpe, Win rates) to CSV files in `data/results/`.

### 4. Compare Results

```bash
# View baseline results
cat data/results/baselines.csv

# View RL vs RL results
cat data/results/rl_vs_rl.csv
```

Or load in Python/Jupyter for analysis:
```python
import pandas as pd

# Compare baselines
baselines = pd.read_csv("data/results/baselines.csv")
print(baselines)

# Compare RL agents
rl_vs_rl = pd.read_csv("data/results/rl_vs_rl.csv")
print(rl_vs_rl)
```

**See [docs/evaluation_guide.md](docs/evaluation_guide.md) for comprehensive evaluation documentation.**

---

## Human Play (CLI)

Play against the market or a random opponent.

```bash
# Single Player
python -m mmrl.human.cli_play --mode single --events on --impact on

# Two Player vs Random
python -m mmrl.human.cli_play --mode two --opponent random
```

---

## Repository Structure

- `src/mmrl/env`: Gym environments (Single/Two Player) & Core logic (Cards, Quotes, Liquidity).
- `src/mmrl/agents`: RL implementations (DQN, IPPO, MAPPO).
- `src/mmrl/baselines`: Simple policies (Random, EV Oracle, Level-1).
- `src/mmrl/human`: CLI game.
- `src/mmrl/eval`: Metrics and plotting.
- `scripts/`: Training and evaluation runners.
- `data/`: Results, logs, and model checkpoints (auto-created).

---

## Configuration

Configs are located in `src/mmrl/config/`.
- `env.yaml`: Market parameters (sigma, spread, liquidity, events).
- `dqn.yaml`, `ippo.yaml`, `mappo.yaml`: Algorithm hyperparameters.

---

## Testing

```bash
pytest
```

---

## License

MIT
