# Baseline Comparison: Privileged vs Realistic

This document explains the key differences between privileged and realistic baselines in the codebase.

## Quick Summary

### EV Baselines

| Feature | EV Oracle | EV Realistic |
|---------|-----------|--------------|
| Fair Value Source | Environment's `info["mu"]` | Computed from observation |
| Knows True Sum | Yes (via `info["true_sum"]`) | No |
| Knows True Depth | Yes (via `info["true_depths"]`) | No |
| Market Impact Handling | Simple (uses known depth) | Bayesian (reasons about hidden depth) |
| Use Case | Upper bound benchmark | Realistic single-agent performance |

### Level-1 Baselines

| Feature | Level-1 (Privileged) | Level-1 Realistic |
|---------|---------------------|-------------------|
| Fair Value Source | Environment's `info["mu"]` | Computed from observation |
| Opponent Modeling | Yes (history tracking) | Yes (history tracking) |
| Impact Reasoning | Uses displayed depth only | Bayesian with uncertainty |
| Risk Adjustment | None | Configurable risk aversion |
| Use Case | Strategic upper bound | Realistic strategic performance |

## EV Oracle (Privileged)

**File**: `src/mmrl/baselines/ev_oracle.py`

The EV Oracle baseline has access to **privileged information** that a real agent would not have:

### What it knows:
1. **`mu`** - The true posterior mean (fair value) computed by the environment
2. **`true_sum`** - The actual sum of the 3 hidden cards
3. **`true_depths`** - The actual liquidity at bid and ask (not just displayed)

### How it decides:
```python
# From info dict (PRIVILEGED):
mu = info["mu"]

# Simple edge calculation:
edge_buy = mu - Y   # Y is ask price
edge_sell = X - mu  # X is bid price

# Act on positive edge:
if edge_buy > 0:
    return max_valid_buy_size
elif edge_sell > 0:
    return max_valid_sell_size
else:
    return PASS
```

### Why it exists:
- Provides a **theoretical upper bound** on performance
- Shows the best possible outcome if you had perfect information
- Useful for measuring how much performance is lost due to information asymmetry

---

## EV Realistic (No Privileged Info)

**File**: `src/mmrl/baselines/ev_realistic.py`

The EV Realistic baseline computes everything from **observable information only**:

### What it observes:
1. **Hints** (indices 10-22 in obs) - Count vector of revealed cards
2. **Event** (indices 23-28 in obs) - One-hot encoded event type
3. **Quote** (indices 0-3 in obs) - Bid, ask, mid, spread
4. **Displayed Depth** (indices 4-5 in obs) - `D_disp = min(L_true, L_cap)`

### What it does NOT know:
- True posterior mean (computes it independently)
- True sum of hidden cards
- True liquidity depth (only sees displayed depth)

### Fair Value Computation:

```python
# Step 1: Decode event and hints from observation
event = decode_event_from_obs(obs)    # Parse one-hot encoding
hints = decode_hints_from_obs(obs)     # Parse count vector

# Step 2: Apply event to filter eligible cards
filtered_deck, value_map = apply_event(FULL_DECK, event)

# Step 3: Compute Bayesian posterior
mu, sigma = posterior_mean_var(hints, filtered_deck, value_map)
```

### Market Impact Handling:

When impact is enabled, the agent must reason about hidden liquidity:

**Case 1: D_disp < L_cap** (Depth is fully observable)
```
L_true = D_disp  (known exactly)
E[(size - L_true)+] = max(0, size - D_disp)
```

**Case 2: D_disp = L_cap** (Depth is capped/truncated)
```
L_true >= L_cap  (truncated distribution)
E[(size - L_true)+ | L_true >= L_cap]  (numerical integration)
```

The expected execution price becomes:
```
E[p_exec_buy] = Y + α × E[(size - L_true)+]
E[p_exec_sell] = X - α × E[(size - L_true)+]
```

And the edge for size `i`:
```
edge_buy(i) = i × (mu - E[p_exec_buy])
edge_sell(i) = i × (E[p_exec_sell] - mu)
```

The agent selects the size that maximizes expected edge.

---

## Mathematical Foundation

### Posterior Mean Calculation

Given:
- `hints` = list of revealed cards (0-3 cards)
- `event` = filter that restricts eligible deck

The posterior mean is:
```
μ = Σ(hints) + (3 - |hints|) × E[X | X ∈ remaining_deck]
```

Where `remaining_deck` is the filtered deck minus the revealed hints.

The variance uses the finite population correction:
```
Var(Sum) = m × Var(Population) × (N - m)/(N - 1)
```
Where `m` = cards remaining to draw, `N` = population size.

### Expected Overflow with Truncated Depth

When `D_disp = L_cap`:
```
E[(i - L)+ | L >= L_cap] = ∫_{L_cap}^{min(i, L_max)} (i - L) × f(L|L>=L_cap) dL
```

Where `f(L|L>=L_cap)` is the truncated lognormal PDF.

---

## Usage Comparison

### EV Oracle
```python
from mmrl.baselines.ev_oracle import EVOracleAgent

agent = EVOracleAgent()
action = agent.act(obs, mask, info=info)  # Uses info["mu"]
```

### EV Realistic
```python
from mmrl.baselines.ev_realistic import EVRealisticAgent

agent = EVRealisticAgent(alpha=0.15)
action = agent.act(obs, mask, info=info)  # Ignores privileged info

# Can also inspect computed fair value:
mu, sigma = agent.get_fair_value(obs)
```

---

## Expected Performance Ranking

In typical evaluations:

**Privileged baselines** (theoretical bounds):
1. **EV Oracle** (highest) - Has perfect information
2. **Level-1** - Strategic with privileged fair value

**Realistic baselines** (achievable performance):
3. **Level-1 Realistic** - Strategic with computed fair value
4. **EV Realistic** - Best single-agent without peeking
5. **Well-trained RL Agent** - Should approach Level-1 Realistic
6. **Random** (lowest) - Baseline floor

### Key Insights

**Information Asymmetry Cost**:
- Gap between EV Oracle and EV Realistic shows cost of not knowing μ
- Gap between Level-1 and Level-1 Realistic shows same for strategic play

**RL Agent Targets**:
- In single-player: Match EV Realistic
- In two-player: Match Level-1 Realistic (which accounts for opponent)

**Warning Signs**:
- RL agent beats realistic baselines significantly → check for bugs
- RL agent beats privileged baselines → definitely a bug or data leak

---

## Level-1 vs Level-1 Realistic

### Level-1 (Privileged)

**File**: `src/mmrl/baselines/level1_crowding.py`

Uses `info["mu"]` for fair value, then applies opponent modeling:
```python
# PRIVILEGED: Gets mu directly
mu = info["mu"]

# Opponent modeling (same as realistic)
opp_demand = estimate_from_history(opponent_history)

# Crowding adjustment
if mu - Y > 0:  # Buy
    for size in range(10, 0, -1):
        overflow = max(0, size + opp_demand - D_disp)
        exec_price = Y + alpha * overflow
        if mu - exec_price > 0:
            return size
```

### Level-1 Realistic (No Privileged Info)

**File**: `src/mmrl/baselines/level1_realistic.py`

Computes fair value from observation, with risk adjustment:
```python
# NO PRIVILEGED INFO: Compute own fair value
mu, sigma = compute_fair_value_from_obs(obs)

# Opponent modeling with uncertainty
opp_demand, opp_uncertainty = estimate_from_history(opponent_history)

# Expected overflow with hidden depth
expected_overflow = compute_expected_overflow(
    size + opp_demand, D_disp, L_cap, liq_cfg
)

# Risk-adjusted edge
exec_price = Y + alpha * expected_overflow
raw_edge = size * (mu - exec_price)
risk_penalty = risk_aversion * (sigma + opp_uncertainty) * size * 0.1
adjusted_edge = raw_edge - risk_penalty
```

### Key Differences

| Aspect | Level-1 | Level-1 Realistic |
|--------|---------|-------------------|
| Fair value | From info | Computed from obs |
| Overflow calc | Uses D_disp directly | E[(size-L)+ \| D_disp] |
| Risk handling | None | Risk aversion param |
| Opponent uncertainty | Not modeled | Explicit in estimate |

---

## Implementation Notes

### EV Realistic Limitations

1. **Remap events**: Cannot decode remap parameters from observation (rare in default config)
2. **Depth model**: Uses approximate lognormal parameters when integrating over truncated depth
3. **Numerical integration**: Uses scipy.integrate.quad for truncated expectations

### Testing

```bash
# Run EV Realistic tests
pytest tests/test_ev_realistic.py -v

# Run Level-1 Realistic tests
pytest tests/test_level1_realistic.py -v

# Compare all baselines
python scripts/eval_baselines.py --n-episodes 100
```

### Configuration

EV Realistic accepts optional configuration:
```python
agent = EVRealisticAgent(
    alpha=0.15,           # Market impact coefficient
    liq_cfg={             # Liquidity model parameters
        "k": 10.0,
        "tau": 0.6,
        "min": 2.0,
        "max": 20.0,
        "display_cap": 10.0
    }
)
```

