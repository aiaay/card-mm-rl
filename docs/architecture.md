# Project Architecture: Card-Sum Market Making RL 

## 0) Purpose & Scope
A reproducible research codebase for studying market-taking strategies in a stylized card-sum market with realistic microstructure. The repository ships:
- A **Gymnasium-style simulator** (single-player and two-player) with **Tier‑2 Liquidity** (displayed vs. hidden depth, slippage/impact).
- **Baselines**: Level‑0 (myopic EV oracle), Level‑1 (EV + crowding-aware adjustment), Random‑Valid.
- **Agents**: DQN → IPPO → MAPPO (parameter sharing & centralized critic variants).
- **Evaluation**: mean P&L, variance, Sharpe/Sortino, max drawdown, valid‑order rate, fill ratio, slippage, probability of ruin, learning curves.
- **Human play** modes: Human vs Random; Human controls 2 agents.
- Feature toggles for events and market impact to enable clean ablations.

---

## 1) Repository Structure
```
card-mm-rl/
├─ README.md
├─ docs/
│  ├─ architecture.md                 # this file
│  └─ design_notes.md                 # extra derivations, formulas, ablations
├─ pyproject.toml                     # or setup.cfg + setup.py
├─ requirements.txt                   # pinned versions for quick start
├─ src/
│  ├─ mmrl/
│  │  ├─ __init__.py
│  │  ├─ config/
│  │  │  ├─ defaults.yaml             # global defaults (seeds, env params)
│  │  │  ├─ env.yaml                  # environment knobs (events, quotes, liquidity, toggles)
│  │  │  ├─ dqn.yaml                  # algo hyperparams and nets
│  │  │  ├─ ippo.yaml
│  │  │  ├─ mappo.yaml
│  │  │  └─ eval.yaml
│  │  ├─ env/
│  │  │  ├─ cards.py                  # ranks, utilities, combinatorics
│  │  │  ├─ events.py                 # market event models and optional persistence
│  │  │  ├─ quotes.py                 # posterior moments, mid or spread or noise
│  │  │  ├─ liquidity.py              # Tier-2 depth draw, display, impact and fills
│  │  │  ├─ execution.py              # order validity, action masking, settlement
│  │  │  ├─ single_env.py             # gym.Env: single-player
│  │  │  ├─ two_player_env.py         # gym.Env: two-player (shared liquidity)
│  │  │  ├─ spaces.py                 # observation or action spaces and encoders
│  │  │  └─ logging_hooks.py          # standardized logging from env.step
│  │  ├─ baselines/
│  │  │  ├─ ev_oracle.py              # Level-0 policy
│  │  │  ├─ level1_crowding.py        # Level-1 crowd-aware policy
│  │  │  └─ random_valid.py           # Random valid action
│  │  ├─ agents/
│  │  │  ├─ common/
│  │  │  │  ├─ nets.py                # MLP or LSTM bodies, heads
│  │  │  │  ├─ replay.py              # prioritized replay for DQN
│  │  │  │  ├─ rollout.py             # vectorized rollout collectors
│  │  │  │  └─ utils.py               # schedules, normalization, masks
│  │  │  ├─ dqn/
│  │  │  │  ├─ dqn_agent.py
│  │  │  │  └─ train_dqn.py
│  │  │  ├─ ippo/
│  │  │  │  ├─ ippo_agent.py          # shared policy, decentralized execution
│  │  │  │  └─ train_ippo.py
│  │  │  └─ mappo/
│  │  │     ├─ mappo_agent.py         # centralized critic
│  │  │     └─ train_mappo.py
│  │  ├─ eval/
│  │  │  ├─ evaluate.py               # metrics over saved policies and baselines
│  │  │  ├─ diagnostics.py            # policy or behavior plots, action histograms
│  │  │  └─ ood_tests.py              # generalization: shifted spreads or liquidity
│  │  ├─ human/
│  │  │  ├─ cli_play.py               # text UI for human vs random or second human
│  │  │  └─ web_demo.py               # optional minimal Gradio app later
│  │  └─ registry.py                  # env factory, config-to-env wiring
│  └─ scripts/
│     ├─ train_all.sh                 # convenience launcher DQN IPPO MAPPO
│     ├─ eval_all.sh
│     └─ make_dataset.py              # rollouts to parquet for offline analysis
├─ tests/
│  ├─ test_env_core.py                # correctness: payoffs, masks, events
│  ├─ test_liquidity_tier2.py         # displayed vs hidden depth; slippage
│  ├─ test_baselines.py               # EV or Level-1 or Random-valid behaviors
│  └─ test_agents_smoke.py            # short trainings converge above random
├─ data/
│  ├─ logs/                           # tensorboard or wandb optional
│  ├─ rollouts/                       # parquet or npz dumps
│  └─ results/                        # csv summaries, plots
└─ LICENSE
```

---

## 2) Game Definition

### 2.1 Cards & Deck
- Ranks: 2..10, J=11, Q=12, K=13, A=14.
- Per round: hidden multiset of **3 cards** without replacement from the **eligible deck** (filtered by event).
- Optional hints: 0–3 revealed cards (subset of those three) added to observation.

### 2.2 Market Events per round
Event set, exhaustive:
1. `remap_value` X to Y this round. Card with rank X is worth Y for this round.
2. `ge10_only`. Only cards worth 10 or more this round.
3. `le7_only`. Only cards worth 7 or less this round.
4. `even_only`. Only even cards this round.
5. `odd_only`. Only odd cards this round.
6. `none`. No constraint.

Toggles and persistence:
- flags.enable_events true or false. If false, force event equals none every round.
- Optional persistence event_persist in 0 to 1. With probability p, repeat last event; else resample from configured frequencies.
- Event frequencies declared in env.yaml when events are enabled.

Events filter the eligible deck or remap a rank’s value before hint sampling and posterior calculation.

### 2.3 Quotes
Let \($S_t$\) be the sum of the 3 cards. Given hints & event:
- Compute posterior moments: mu equals expected S given information, sigma_S equals standard deviation of S given information; \($\mu_t = E[S_t|\cdot]$, $\sigma_{S,t} = \sqrt{Var[S_t|\cdot]}$). Use without-replacement corrections in `quotes.py`.
- Spread rule: spread equals clip s0 plus beta times sigma_S, within s_min and s_max; `spread_t = clip(s0 + beta * sigma_{S,t}, s_min, s_max)`
- Mid rule noisy oracle: mid equals mu plus Gaussian noise with variance sigma_q squared; `mid_t = mu_t + ε_t`, \($ε_t\sim\mathcal{N}(0,\sigma_q^2)$)
- Bid and Ask: X equals mid minus half of spread, Y equals mid plus half of spread; `X_t = mid_t - spread_t/2`, `Y_t = mid_t + spread_t/2`


### 2.4 Liquidity displayed versus hidden
- Nominal depth scale: L_bar equals k divided by sigma_S times spread plus epsilon; \($\bar L_t = \kappa / (\sigma_{S,t} (\text{spread}_t + \varepsilon))$\)
- **True depth** (per side) drawn lognormally: `L_true ~ LogNormal(mean=ln(\bar L_t) - 0.5 τ^2, sigma=τ)`; clip to `[L_min, L_max]`.
- **Displayed depth**: `D_disp = min(L_true, L_cap)` is **observable** in the state.

### 2.5 Execution and Impact no pro-rata fills
- Aggregate taker demand per side, e.g., total buy size $q_{buy} = i_1 + i_2$ in two‑player.
- Toggleable impact:
  - flags.enable_impact true or false. If false, all fills execute at quoted prices Y for buys, X for sells regardless of overflow. Use this for ablations.
  - If true and $q_{buy} ≤ L_{true}$: all buy fills @ `Y_t`.
  - If true and overflow q_ov equals q_buy minus true depth and is greater than zero: linear impact with a single executed price for all buyers: `p_exec_buy = Y_t + α * q_ov`. Symmetric for sells with p_exec_sell equals `X_t` minus alpha times q_ov.

### 2.6 Valid Orders and Budget
- Initial budget: `W0 = 500` per agent.
- **Long affordability:** `i * Y_t ≤ W_t`.
- **Short worst‑case loss constraint:** `i * (S_max(event) − X_t) ≤ W_t`. Here `S_max` depends on current event filter.
- Invalid actions masked out in the policy.

### 2.7 Rewards and Settlement
- After execution, reveal $S_t$.
- Buy P&L: $r_t = i (S_t - p^{buy}_{exec})$. Sell P&L: $r_t = i (p^{sell}_{exec} - S_t)$. Pass: 0.
- Update budget to $W_{t+1}$; optional stop‑out if $W_t$ falls below threshold.
- Episode length: 10 rounds.

### 2.8 Single vs Two-Player
- **Single:** one taker vs. exogenous quotes/liquidity.
- **Two‑Player:** two takers share the same quote & liquidity, act simultaneously. Observations include opponent last action/size and recent slippage/fill ratios (to support opponent modeling).

---

## 3) Observation, Action, Reward OAR

### 3.1 Observation per agent
- Quote: `X_t`, `Y_t`, `mid_t`, `spread_t`.
- Liquidity: `D_disp_bid`, `D_disp_ask` (displayed depth); last‑round slippage & fill ratio features.
- Events: one‑hot vector for current event (or belief if hidden/persistent).
- Hints: 13‑dim count vector of visible ranks (2..A).
- Bankroll & time: `W_t/W0`, `t/T`.
- Opponent (two‑player): last action side ∈ {−1,0,+1}, last size ∈ [0,10].
- Flags for clarity in logs: enable_events, enable_impact.

### 3.2 Action Space
- Discrete 21 actions: `{Pass} ∪ {Buy 1..10} ∪ {Sell 1..10}`.
- Action masking applied using validity constraints.

### 3.3 Reward
- Default per‑round realized P&L. Optionally use `log(W_{t+1})−log(W_t)` for risk‑aware training.

---

## 4) Baselines

### 4.1 Level-0 Myopic EV Oracle
- Compute $\mu_t$. Directional rule: buy if $\mu_t − Y_t > 0$, sell if $X_t − \mu_t > 0$, else pass.
- Sizing: (a) max valid size; (b) fraction‑Kelly using per‑unit mean/variance; or (c) fixed small size.

### 4.2 Level-1 Crowding-Aware
- Assume opponent plays Level‑0 with known probability `p_hit_side` estimated from last‑k rounds.
- Predict expected overflow vs displayed depth; if expected overflow > 0, downsize or pass to avoid impact.

### 4.3 Random-Valid
- Uniform draw from valid actions after masking.

All baselines exposed via a common policy API for fair evaluation.

---

## 5) Agents and Training

### 5.1 DQN Single-Agent
- MLP or LSTM backbone; optional dueling heads.
- Prioritized replay; target networks; epsilon-greedy with masked actions.

### 5.2 IPPO Two-Agent Shared Policy
- Decentralized execution, shared parameters. Entropy bonus, GAE, clipping PPO.
- Recurrent variant recommended if events are persistent.

### 5.3 MAPPO Centralized Critic
- Actor observes its own observation; critic sees joint observations or actions or concatenations for stability.

### 5.4 Common Training Utilities
- Vectorized envs; reward or observation normalization; learning-rate and entropy schedules; checkpointing; seeding.

---

## 6) Evaluation and Diagnostics

### 6.1 Metrics
- **Return:** mean P&L per episode, variance; Sharpe/Sortino; max drawdown; probability of ruin.
- **Execution:** valid‑order rate; fill ratio; mean slippage; impact per unit; rejection rate (invalid attempts if not masked in baseline tests).
- **Learning:** moving‑window returns, stability across seeds; sample efficiency (#steps to beat Random‑Valid and Level‑0).

### 6.2 Protocol
- Fixed train/val/test seeds and **out‑of‑distribution (OOD)** test with shifted `spread`, `α`, `τ`, and event frequencies.
- Report mean ± 95% CI over ≥5 seeds.

### 6.3 Policy Diagnostics
- Action histograms versus edge magnitude; size versus displayed depth; sensitivity to uncertainty; opponent-response curves; ablations events off versus on, impact off versus on, known versus partially known depth, no persistence versus persistence.

---

## 7) Human Play

### 7.1 CLI Text UI
- `python -m mmrl.human.cli_play --mode single|two --opponent random_valid`.
- Shows quote, displayed depth, events, revealed hints, current bankroll; user picks action by number; engine prints execution price, S, and P&L.

### 7.2 Two-Human Mode
- Alternating prompts for Agent A and Agent B; resolves simultaneous actions after both inputs are captured.

Optional Web front end in human or web_demo.py.

---

## 8) Configuration System
- YAML configs in `src/mmrl/config/` (Hydra‑style or plain argparse loaders).
- - Hierarchical overrides: environment (`env.yaml`), algorithm (`dqn.yaml`, `ippo.yaml`, `mappo.yaml`), evaluation (`eval.yaml`).
- Key toggles:
  - flags.enable_events true
  - flags.enable_impact true
  - event_persist 0.7 optional persistence
- Example launch:
```
python -m mmrl.agents.ippo.train_ippo \
  env.event_persist=0.7 env.alpha=0.3 env.spread.beta=0.25 \
  train.total_steps=2_000_000 seed=123
```

---

## 9) Key APIs Sketches

### 9.1 Env Factory (`registry.py`)
```python
from mmrl.env.single_env import SingleCardEnv
from mmrl.env.two_player_env import TwoPlayerCardEnv

def make_env(kind: str, cfg):
    if kind == "single":
        return SingleCardEnv(cfg)
    elif kind == "two":
        return TwoPlayerCardEnv(cfg)
    raise ValueError(kind)
```

### 9.2 Step Output Contract Two-Player
```python
obs = (obs_a, obs_b)
obs, (r_a, r_b), terminated, truncated, info = env.step((act_a, act_b))
# info contains per-agent exec_price, fill_ratio, slippage, mu_t, sigma_S_t, S_t, masks, flags
```

### 9.3 Liquidity Tier-2 liquidity.py
```python
def draw_true_depth(sigma_S, spread, k, tau, caps):
    L_bar = k / (sigma_S * (spread + 1e-8))
    L = lognormal(L_bar, tau)
    return clip(L, caps.min, caps.max)

def displayed_depth(L_true, cap):
    return min(L_true, cap)

def exec_price_buy(Y, q_total, L_true, alpha, enable_impact=True):
    if not enable_impact: return Y
    overflow = max(0.0, q_total - L_true)
    return Y + alpha * overflow

def exec_price_sell(X, q_total, L_true, alpha, enable_impact=True):
    if not enable_impact: return X
    overflow = max(0.0, q_total - L_true)
    return X - alpha * overflow
```

### 9.4 EV Oracle Level-0
```python
def ev_action(mu, x_bid, y_ask, W, smax, sizing="max"):
    edge_buy  = mu - y_ask
    edge_sell = x_bid - mu
    if max(edge_buy, edge_sell) <= 0: return PASS
    side = BUY if edge_buy > edge_sell else SELL
    # mask by affordability / short worst-case
    i_max = feasible_size(side, W, x_bid, y_ask, smax)
    i = size_rule(sizing, i_max, edge=max(edge_buy, edge_sell))
    return (side, i)
```

---

## 10) Testing Strategy
- **Unit tests** for: posterior moments; event filtering; validity masks; payoff formulas; liquidity & displayed vs hidden; impact pricing; reproducible seeding.
- **Integration tests**: short runs where EV oracle ≥ Random‑Valid; IPPO beats Random‑Valid after N steps.
- **Human tests**: CLI scripts to verify logical flows.

---

## 11) Reproducibility & Logging
- Global seed control (env RNG, PyTorch, numpy, Python).
- Structured logs (TensorBoard/CSV); per‑step `env_log` parquet via `scripts/make_dataset.py` for offline analysis.
- Save configs with checkpoints; store git commit hash in run metadata.

---

## 12) Default Hyperparameters Suggested
- Quotes: `sigma_q=0.5`; spread: `s0=0.8, beta=0.25, [1.0, 3.0]`
- Liquidity: `k` s.t. median depth ≈ 6–8; `τ=0.6`; `L_min=2, L_max=20`; display cap `L_cap=10`
- Impact: `α=0.3` price points / overflow unit
- Events frequencies when enabled: `none 0.6`, `even_only 0.1`, `ge10_only 0.1`, `le7_only 0.1`, `odd_only 0.1`. `remap_value` can be a rare override for example one to two percent. Optional event_persist equals 0.7.
- Episode: 10 rounds; `W0=500`; stop‑out optional at `0.2 * W0`.

---

## 13) Roadmap Optional Extensions
- Endogenous market maker third agent selecting quotes and displayed depth.
- Inventory carry-over and risk penalties across rounds.
- Partial observability of events with belief states.
- Distributional RL heads; constrained RL with ruin probability constraints.
- Web demo with replay viewer per-round cards, quotes, fills, P and L.

---

## 14) Quickstart Recipes
- **Baseline eval:** `python -m mmrl.eval.evaluate --policy ev_oracle --episodes 10000 --events on --impact on`
- **Train DQN single:** `python -m mmrl.agents.dqn.train_dqn env.kind=single env.flags.enable_events=on env.flags.enable_impact=off train.total_steps=1000000`
- **Train IPPO two:** `python -m mmrl.agents.ippo.train_ippo env.kind=two env.flags.enable_events=on env.flags.enable_impact=on train.total_steps=2000000`
- **Train MAPPO:** `python -m mmrl.agents.mappo.train_mappo env.kind=two env.flags.enable_events=on env.flags.enable_impact=on train.total_steps=2000000`
- **Human vs Random:** `python -m mmrl.human.cli_play --mode two --opponent random_valid --events on --impact off`

---

## 15) License and Citation
- MIT License

