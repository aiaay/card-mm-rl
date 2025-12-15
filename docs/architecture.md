# Project Architecture: Card-Sum Market Making RL

## 0) Purpose & Scope

A reproducible research codebase for studying market-taking strategies in a stylized card-sum market with realistic microstructure. The repository provides:

- **Gymnasium-style simulators**: Single-player (`SingleCardEnv`) and two-player (`TwoPlayerCardEnv`) environments with Tier-2 liquidity (displayed vs. hidden depth, slippage/impact).
- **Baselines**: Random-Valid, Level-0 EV Oracle (myopic), Level-1 Crowding-Aware.
- **RL Agents**: DQN (single-player), IPPO (two-player, shared policy), MAPPO (two-player, centralized critic).
- **Evaluation**: Mean P&L, variance, Sharpe ratio, ruin probability, valid-order rate, slippage tracking. 
- **Human Play**: CLI interface for single-player or two-player (vs random opponent).
- **Feature Toggles**: Events and market impact can be enabled/disabled for clean ablations.

---

## 1) Repository Structure

```
card-mm-rl/
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── docs/
│   ├── architecture.md              # This file
│   ├── task.md                      # Development task breakdown
│   └── evaluation_guide.md          # Evaluation documentation
├── src/mmrl/
│   ├── __init__.py
│   ├── config/
│   │   ├── loader.py                # Config loading utilities
│   │   ├── defaults.yaml            # Global defaults (seed, device, paths)
│   │   ├── env.yaml                 # Environment parameters
│   │   ├── dqn.yaml                 # DQN hyperparameters
│   │   ├── ippo.yaml                # IPPO hyperparameters
│   │   ├── mappo.yaml               # MAPPO hyperparameters
│   │   └── eval.yaml                # Evaluation settings
│   ├── env/
│   │   ├── cards.py                 # Deck, ranks, drawing, sum calculation
│   │   ├── events.py                # Event types, sampling, deck filtering
│   │   ├── quotes.py                # Posterior moments, quote generation
│   │   ├── liquidity.py             # Tier-2 depth, display cap, impact pricing
│   │   ├── execution.py             # Action masking, validity constraints
│   │   ├── spaces.py                # Observation/action space definitions
│   │   ├── single_env.py            # Single-player Gymnasium environment
│   │   ├── two_player_env.py        # Two-player Gymnasium environment
│   │   └── logging_hooks.py         # Structured logging utilities
│   ├── baselines/
│   │   ├── random_valid.py          # Random valid action policy
│   │   ├── ev_oracle.py             # Level-0 EV oracle policy (privileged)
│   │   ├── ev_realistic.py          # Bayesian baseline (no privileged info)
│   │   ├── level1_crowding.py       # Level-1 crowd-aware policy (privileged)
│   │   └── level1_realistic.py      # Level-1 realistic (no privileged info)
│   ├── agents/
│   │   ├── common/
│   │   │   ├── nets.py              # MLP neural network module
│   │   │   ├── replay.py            # Experience replay buffer (DQN)
│   │   │   ├── rollout.py           # Rollout buffer (PPO)
│   │   │   └── utils.py             # Utility functions
│   │   ├── dqn/
│   │   │   ├── dqn_agent.py         # DQN agent implementation
│   │   │   └── train_dqn.py         # DQN training script
│   │   ├── ippo/
│   │   │   ├── ippo_agent.py        # IPPO agent (shared policy)
│   │   │   └── train_ippo.py        # IPPO training script
│   │   └── mappo/
│   │       ├── mappo_agent.py       # MAPPO agent (centralized critic)
│   │       └── train_mappo.py       # MAPPO training script
│   ├── eval/
│   │   ├── evaluate.py              # Policy evaluation utilities
│   │   ├── compare_agents.py        # Agent comparison framework
│   │   ├── diagnostics.py           # Behavior analysis tools
│   │   └── ood_tests.py             # Out-of-distribution testing
│   └── human/
│       ├── cli_play.py              # Command-line human play interface
│       └── web_demo.py              # (Placeholder) Web demo
├── scripts/
│   ├── train_all.sh                 # Train all agents sequentially
│   ├── eval_baselines.py            # Evaluate baseline policies
│   ├── eval_all.py                  # Evaluate all agents
│   ├── eval_rl_vs_rl.py             # RL vs RL matchups
│   └── run_all_evaluations.sh       # Run complete evaluation suite
├── tests/                           # Comprehensive test suite
│   ├── test_cards.py
│   ├── test_events.py
│   ├── test_quotes.py
│   ├── test_liquidity_*.py
│   ├── test_single_*.py
│   ├── test_two_*.py
│   ├── test_baselines.py
│   ├── test_agents_smoke.py
│   └── ...
└── data/
    ├── logs/                        # TensorBoard logs
    ├── models/                      # Saved model checkpoints
    └── results/                     # Evaluation CSV outputs
```

---

## 2) Game Definition

### 2.1 Cards & Deck

- **Ranks**: 2, 3, 4, 5, 6, 7, 8, 9, 10, J(11), Q(12), K(13), A(14)
- **Full Deck**: Standard 52-card deck (4 suits × 13 ranks)
- **Per Round**: 3 cards drawn without replacement from the eligible deck
- **Sum (S)**: The sum of the 3 drawn card values (the "true value" to trade around)

**Implementation** (`cards.py`):
```python
ALL_RANKS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
FULL_DECK = sorted(ALL_RANKS * 4)  # 52 cards

def draw_cards(deck, n, rng) -> (drawn, remaining)
def calculate_sum(cards) -> int
```

### 2.2 Market Events

Events filter the eligible deck or remap card values each round:

| Event Type | Description |
|------------|-------------|
| `none` | No constraint (full deck available) |
| `ge10_only` | Only cards with value ≥ 10 |
| `le7_only` | Only cards with value ≤ 7 |
| `even_only` | Only even-valued cards |
| `odd_only` | Only odd-valued cards |
| `remap_value` | Remap rank X to value Y for this round |

**Configuration** (`env.yaml`):
```yaml
flags:
  enable_events: true   # Toggle events on/off

events:
  freq:
    none: 0.6
    ge10_only: 0.1
    le7_only: 0.1
    even_only: 0.1
    odd_only: 0.1
    remap_value: 0.0    # Rare special event
  persistence: 0.7      # Probability to repeat last event
```

**Implementation** (`events.py`):
- `Event` dataclass with `type` and `params`
- `apply_event(deck, event)` → (filtered_deck, value_map)
- `sample_event(rng, cfg, last_event)` → Event

### 2.3 Hints

After drawing 3 cards, 0–3 may be revealed as "hints" to the agent:

**Configuration**:
```yaml
hints:
  count: 2              # Fixed count, or omit for random
  probabilities: [0.35, 0.30, 0.20, 0.15]  # P(0), P(1), P(2), P(3) hints
```

### 2.4 Quotes (Price Formation)

Given hints and event, compute posterior moments and generate quotes:

**Posterior Calculation** (`quotes.py`):
```python
def posterior_mean_var(hints, filtered_deck, value_map) -> (mu, sigma)
```
- Uses finite population sampling without replacement
- Variance correction factor: `(N - m) / (N - 1)` where m = cards remaining to draw

**Quote Generation**:
```
mid = mu + ε,  where ε ~ N(0, σ_q²)
spread = clip(s0 + β × σ_S, s_min, s_max)
bid (X) = mid - spread/2
ask (Y) = mid + spread/2
```

**Default Parameters**:
- `sigma_q = 0.5` (quote noise)
- `spread.s0 = 0.8`, `spread.beta = 0.25`
- `spread.min = 1.0`, `spread.max = 3.0`

### 2.5 Tier-2 Liquidity

**True Depth** (hidden):
```
L_bar = k / (σ_S × (spread + ε))
L_true ~ LogNormal(ln(L_bar) - 0.5τ², τ)
L_true = clip(L_true, L_min, L_max)
```

**Displayed Depth** (observable):
```
D_disp = min(L_true, L_cap)
```

**Default Parameters**:
- `k = 10.0` (depth scaling)
- `tau = 0.6` (lognormal volatility)
- `L_min = 2.0`, `L_max = 20.0`
- `L_cap = 10.0` (display cap)

### 2.6 Execution & Market Impact

**Impact Model** (`liquidity.py`):
```python
def exec_price_buy(Y, q_total, L_true, alpha, enable_impact):
    if not enable_impact:
        return Y
    overflow = max(0, q_total - L_true)
    return Y + alpha * overflow

def exec_price_sell(X, q_total, L_true, alpha, enable_impact):
    # Symmetric for sells
```

- `alpha = 0.15` (impact coefficient, gentler during training)
- Impact is **toggleable** via `flags.enable_impact`

### 2.7 Valid Orders & Budget Constraints

**Initial Budget**: `W0 = 500`

**Action Validity** (`execution.py`):
- **Buy constraint**: `size × Y ≤ W` (can afford to buy)
- **Sell constraint**: `size × (S_max - X) ≤ W` (worst-case loss covered)

Where `S_max` is the maximum possible sum given the current event filter.

**Pass Limit**: After `max_consecutive_passes` (default: 3), Pass is masked if other actions are available.

### 2.8 Rewards & Settlement

**Per-Round P&L**:
```
Buy:  reward = size × (S - p_exec_buy)
Sell: reward = size × (p_exec_sell - S)
Pass: reward = -pass_penalty × (1 + 0.5 × (consecutive_passes - 1))
```

**Budget Update**: `W_{t+1} = W_t + reward`

**Termination**:
- Episode length reached (`T = 10` or `20`)
- Ruin: `W < stop_out × W0` (default `stop_out = 0.1`)

---

## 3) Observation, Action, Reward (OAR)

### 3.1 Observation Space

**Dimension**: 35-element float vector

| Component | Indices | Description |
|-----------|---------|-------------|
| Quote | 0-3 | bid (X), ask (Y), mid, spread |
| Displayed Depth | 4-5 | bid depth, ask depth |
| Last-Step Metrics | 6-9 | slippage_bid, slippage_ask, fill_bid, fill_ask |
| Hints | 10-22 | 13-dim count vector (ranks 2..14) |
| Event | 23-28 | 6-dim one-hot (none, ge10, le7, even, odd, remap) |
| State | 29-30 | W/W0, t/T (normalized wealth and time) |
| Flags | 31-32 | enable_events, enable_impact |
| Opponent | 33-34 | last action side (-1,0,1), last size |

**Implementation** (`spaces.py`):
```python
def get_obs_shape() -> (35,)
def build_obs(quote, depths, metrics, hints, event, state, flags, opponent_last) -> np.array
```

### 3.2 Action Space

**Discrete(21)**:
- `0`: Pass
- `1-10`: Buy size 1-10
- `11-20`: Sell size 1-10

**Masking**: Invalid actions are masked before policy selection.

### 3.3 Reward

Default: Per-round realized P&L (with optional pass penalty).

---

## 4) Environments

### 4.1 SingleCardEnv

**Single-player** market-taking against exogenous quotes/liquidity.

```python
from mmrl.env.single_env import SingleCardEnv

env = SingleCardEnv(cfg)
obs, info = env.reset(seed=42)
obs, reward, terminated, truncated, info = env.step(action)
```

**Info Dict** (on reset/step):
- `mu`, `sigma`: Posterior moments (privileged for baselines)
- `event`: Current event dict
- `mask`: Valid action mask
- `true_sum`, `true_depths`: Hidden ground truth (for logging)

### 4.2 TwoPlayerCardEnv

**Two-player** environment with shared liquidity and simultaneous actions.

```python
from mmrl.env.two_player_env import TwoPlayerCardEnv

env = TwoPlayerCardEnv(cfg)
(obs_a, obs_b), info = env.reset(seed=42)
(obs_a, obs_b), (r_a, r_b), term, trunc, info = env.step((act_a, act_b))
```

**Key Differences**:
- Shared quote and liquidity per round
- Aggregate demand determines impact: `q_buy = size_a + size_b`
- Each agent sees opponent's last action in observation
- Both agents share the same hints

---

## 5) Baselines

### 5.1 Random-Valid (`random_valid.py`)

Uniformly samples from valid actions.

```python
class RandomValidAgent:
    def act(self, obs, mask, eval_mode=True) -> int
```

### 5.2 EV Oracle Level-0 (`ev_oracle.py`)

Uses **privileged access** to posterior mean `mu`:
- Buy max valid size if `mu - Y > 0` and edge is positive
- Sell max valid size if `X - mu > 0` and edge is larger
- Otherwise Pass

⚠️ **Note**: This baseline "cheats" by knowing the true fair value from the environment.

```python
class EVOracleAgent:
    def act(self, obs, mask, info=None, eval_mode=True) -> int
```

### 5.3 EV Realistic (`ev_realistic.py`)

**No privileged information**. Computes fair value using only observable data:
- Decodes hints and event from observation vector
- Computes Bayesian posterior `E[S | hints, event]` using standard probability
- Handles market impact by computing expected overflow given displayed depth

**Without impact**:
```
edge = max(mu - Y, X - mu)
```
Buy if `mu - Y > 0` is larger, Sell if `X - mu > 0` is larger.

**With impact** (Tier-2 liquidity):
```
E[p_exec_buy] = Y + α × E[(size - L_true)+ | D_disp]
edge_buy(i) = i × (mu - E[p_exec_buy])
```

Selects size that maximizes expected edge while respecting budget constraints.

**Key Feature**: Uses `D_disp = min(L_true, L_cap)` to reason about hidden liquidity:
- If `D_disp < L_cap`: True depth is known exactly (`L_true = D_disp`)
- If `D_disp = L_cap`: True depth is uncertain (`L_true >= L_cap`), integrates over truncated distribution

```python
class EVRealisticAgent:
    def __init__(self, alpha=0.15, liq_cfg=None)
    def act(self, obs, mask, info=None, eval_mode=True) -> int
    def get_fair_value(self, obs) -> (mu, sigma)
```

### 5.4 Level-1 Crowding-Aware (`level1_crowding.py`) - Privileged

Uses **privileged access** to `mu` plus opponent modeling:
- Tracks opponent action history
- Estimates expected opponent demand on same side
- Adjusts size to avoid overflow/impact

⚠️ **Note**: Uses `info["mu"]` (privileged).

```python
class Level1Policy:
    def __init__(self, history_len=10, alpha=0.3)
    def update(self, opp_side, opp_size)  # Call after each round
    def act(self, obs, mask, info=None) -> int
```

### 5.5 Level-1 Realistic (`level1_realistic.py`) - No Privileged Info

**No privileged information**. Combines:
- Bayesian fair value estimation (like EV Realistic)
- Opponent modeling through action history tracking
- Expected impact calculation with uncertain liquidity
- Risk-adjusted edge computation

**Key Features**:
- Computes fair value from observable data only
- Tracks opponent history to estimate crowding probability
- Uses expected overflow calculation for hidden depth
- Includes risk aversion parameter for uncertainty handling

```python
class Level1RealisticPolicy:
    def __init__(self, history_len=10, alpha=0.15, risk_aversion=0.5)
    def update(self, opp_side, opp_size)  # Track opponent
    def reset()  # Clear history (episode start)
    def act(self, obs, mask, info=None) -> int
    def get_fair_value(obs) -> (mu, sigma)
    def get_opponent_estimate(side) -> (demand, uncertainty)
```

---

## 5.6 Baseline Summary Table

| Baseline | Privileged Info | Opponent Model | Use Case |
|----------|-----------------|----------------|----------|
| Random | None | No | Lower bound |
| EV Oracle | mu, true_depths | No | Upper bound |
| EV Realistic | None | No | Realistic single-agent bound |
| Level-1 | mu | Yes | Strategic upper bound |
| Level-1 Realistic | None | Yes | Realistic strategic bound |

---

## 6) RL Agents

### 6.1 DQN (`agents/dqn/`)

**Architecture**: MLP Q-network with action masking

```python
class DQNAgent:
    def __init__(self, obs_dim, action_dim, cfg)
    def act(self, obs, mask, eval_mode=False) -> int
    def step(self, obs, action, reward, next_obs, done, mask)
    def update(self) -> loss
```

**Features**:
- Experience replay buffer
- Target network with periodic updates
- Epsilon-greedy exploration with decay
- Huber loss

**Training** (`train_dqn.py`):
```bash
python -m mmrl.agents.dqn.train_dqn --episodes 1000 --seed 42
```

### 6.2 IPPO (`agents/ippo/`)

**Independent PPO** for two-player with **shared parameters**.

```python
class IPPOAgent:
    def __init__(self, obs_dim, action_dim, cfg)
    def step(self, obs, mask) -> (action, log_prob, value)
    def store(self, obs, action, reward, log_prob, val, done, mask)
    def update(self, last_val) -> {"loss", "kl", "entropy"}
```

**Architecture**: Separate actor and critic MLPs (but shared weights across agents)

**Features**:
- GAE (Generalized Advantage Estimation)
- PPO clipping
- Entropy bonus for exploration
- Early stopping on KL divergence

**Training** (`train_ippo.py`):
```bash
python -m mmrl.agents.ippo.train_ippo --steps 50000 --seed 42
```

### 6.3 MAPPO (`agents/mappo/`)

**Multi-Agent PPO** with **centralized critic**.

```python
class MAPPOAgent:
    def __init__(self, obs_dim, joint_obs_dim, action_dim, cfg)
    def step(self, obs, joint_obs, mask) -> (action, log_prob, value)
    def store(self, obs, joint_obs, action, reward, log_prob, val, done, mask)
    def update(self, last_val) -> loss
```

**Key Difference**: Critic receives `joint_obs = concat(obs_a, obs_b)` for better value estimation.

**Training** (`train_mappo.py`):
```bash
python -m mmrl.agents.mappo.train_mappo --steps 50000 --seed 42
```

### 6.4 Common Components

**MLP Network** (`common/nets.py`):
```python
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim)
```

**Replay Buffer** (`common/replay.py`): For DQN experience replay

**Rollout Buffer** (`common/rollout.py`): For PPO trajectory collection with GAE

---

## 7) Evaluation

### 7.1 Metrics

| Metric | Description |
|--------|-------------|
| `return_mean` | Average episode return (P&L) |
| `return_std` | Standard deviation of returns |
| `sharpe` | `mean / (std + ε)` (simplified Sharpe) |
| `ruin_prob` | Fraction of episodes ending in ruin |
| `valid_rate` | Fraction of valid actions taken |
| `pass_rate` | Fraction of Pass actions |
| `mean_slippage` | Average execution slippage |

### 7.2 Evaluation Scripts

**Baseline Evaluation**:
```bash
python scripts/eval_baselines.py --n-episodes 100
# Output: data/results/baselines.csv
```

**RL Agent Evaluation**:
```bash
python scripts/eval_all.py --n-episodes 100
# Output: data/results/eval_all.csv
```

**RL vs RL Matchups**:
```bash
python scripts/eval_rl_vs_rl.py --agents ippo mappo --episodes 100
# Output: data/results/rl_vs_rl.csv
```

**Agent Comparison**:
```bash
python -m mmrl.eval.compare_agents --agent ippo --opponent mappo --episodes 100
```

### 7.3 TensorBoard Logging

Training logs are saved to `data/logs/{dqn,ippo,mappo}/`:
```bash
tensorboard --logdir data/logs
```

Logged metrics:
- `train/episode_return`: Per-episode return
- `train/epsilon`: Exploration rate (DQN)
- `train/loss`, `train/kl`, `train/entropy`: PPO losses

---

## 8) Human Play

### 8.1 CLI Interface (`human/cli_play.py`)

```bash
# Single player
python -m mmrl.human.cli_play --mode single --events on --impact on

# Two player vs random opponent
python -m mmrl.human.cli_play --mode two --opponent random --events on --impact off
```

**Display**:
- Current round, event type
- Revealed hints
- Quote (bid @ ask)
- Displayed depth
- Current wallet
- Valid actions

**Post-Action**:
- Execution price
- True sum reveal
- P&L
- Price impact (if any)

---

## 9) Configuration System

### 9.1 YAML Configs

Located in `src/mmrl/config/`:

**`defaults.yaml`**:
```yaml
seed: 42
device: "cpu"
output_dir: "data/results"
log_dir: "data/logs"
```

**`env.yaml`**:
```yaml
kind: "single"          # or "two"
episode_length: 20
W0: 500.0
stop_out: 0.1

sigma_q: 0.5
spread: {s0: 0.8, beta: 0.25, min: 1.0, max: 3.0}
liquidity: {k: 10.0, tau: 0.6, min: 2.0, max: 20.0, cap: 10.0}
alpha: 0.15
pass_penalty: 0.05
max_consecutive_passes: 3

flags:
  enable_events: true
  enable_impact: false

events:
  freq: {none: 0.6, ge10_only: 0.1, le7_only: 0.1, even_only: 0.1, odd_only: 0.1}
  persistence: 0.7
```

### 9.2 Config Access Pattern

Configs are passed as dictionaries and accessed via helper:
```python
def _get_cfg(self, key, default=None):
    # Supports dot notation: "flags.enable_events"
```

---

## 10) Testing Strategy

### 10.1 Test Categories

| Category | Files | Coverage |
|----------|-------|----------|
| Core Mechanics | `test_cards.py`, `test_quotes.py`, `test_events.py` | Deck, posterior, events |
| Liquidity | `test_liquidity_*.py`, `test_impact.py` | Depth, display, impact |
| Environments | `test_single_*.py`, `test_two_*.py` | Reset, step, termination |
| Baselines | `test_baseline_*.py` | Policy behaviors |
| Agents | `test_agents_smoke.py` | Training convergence |
| Evaluation | `test_eval_*.py`, `test_ood.py` | Metrics, OOD tests |

### 10.2 Running Tests

```bash
pytest                          # All tests
pytest tests/test_cards.py      # Specific file
pytest -v                       # Verbose
```

---

## 11) Reproducibility

### 11.1 Seeding

- Environment: `env.reset(seed=42)`
- NumPy RNG: `np.random.RandomState(seed)`
- PyTorch: Set via config

### 11.2 Checkpointing

Models saved to `data/models/{agent}/`:
- DQN: `dqn_ep{N}.pt`, `dqn_final.pt`
- IPPO: `ippo_step{N}.pt`, `ippo_final.pt`
- MAPPO: `mappo_step{N}.pt`, `mappo_final.pt`

Checkpoint contents:
```python
{
    'q_net': state_dict,       # DQN
    'ac': state_dict,          # PPO
    'optimizer': state_dict,
    'episode': N,
    'epsilon': float,          # DQN only
}
```

---

## 12) Default Hyperparameters

### Environment
- Episode length: 20 (training), 10 (eval)
- Initial wealth: W0 = 500
- Stop-out: 10% of W0
- Pass penalty: 0.05

### DQN
- Hidden dims: [64, 64]
- Learning rate: 1e-3
- Batch size: 64
- Buffer size: 20,000
- Epsilon: 1.0 → 0.1 (decay 0.999)
- Target update: every 200 steps
- Gamma: 0.99

### IPPO/MAPPO
- Hidden dims: [64, 64]
- Learning rate: 3e-4
- Rollout steps: 1024
- Train iterations: 6
- Clip ratio: 0.2
- Entropy coef: 0.02
- Value coef: 0.5
- GAE lambda: 0.95
- Gamma: 0.99

---

## 13) Quickstart Recipes

### Train All Agents
```bash
# DQN (single-player)
python -m mmrl.agents.dqn.train_dqn --episodes 1000 --seed 42

# IPPO (two-player)
python -m mmrl.agents.ippo.train_ippo --steps 50000 --seed 42

# MAPPO (two-player)
python -m mmrl.agents.mappo.train_mappo --steps 50000 --seed 42

# Or all at once
bash scripts/train_all.sh
```

### Evaluate
```bash
# Baselines
python scripts/eval_baselines.py --n-episodes 100

# All agents
python scripts/eval_all.py --n-episodes 100

# RL vs RL
python scripts/eval_rl_vs_rl.py --agents ippo mappo --episodes 100

# View training curves
tensorboard --logdir data/logs
```

### Human Play
```bash
python -m mmrl.human.cli_play --mode single --events on --impact on
python -m mmrl.human.cli_play --mode two --opponent random
```

```bash
streamlit run src/mmrl/human/web_demo.py
```

---

## 14) Future Extensions

- Endogenous market maker (third agent setting quotes)
- Inventory carry-over across rounds
- Partial observability of events (belief states)
- Distributional RL / risk-constrained RL
- Web demo with replay viewer

---

## 15) License

MIT License
