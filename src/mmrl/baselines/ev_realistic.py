"""
EV Realistic Baseline

A Bayesian-optimal baseline that uses ONLY observable information:
- Quote (bid/ask)
- Event (one-hot encoded in obs)
- Hints (revealed cards)
- Displayed depth (NOT true depth)

NO peeking at:
- True sum of hidden cards
- True liquidity depth
- Posterior moments computed by the environment

This baseline computes its own posterior fair value from observable data
and accounts for expected market impact when impact is enabled.
"""

import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from scipy import stats
from scipy.integrate import quad

from mmrl.env.cards import ALL_RANKS, FULL_DECK
from mmrl.env.events import (
    Event, EVENT_NONE, EVENT_GE10, EVENT_LE7, 
    EVENT_EVEN, EVENT_ODD, EVENT_REMAP_VALUE, 
    apply_event, get_default_value_map
)
from mmrl.env.quotes import posterior_mean_var


# Default liquidity config (should match env defaults)
DEFAULT_LIQUIDITY_CFG = {
    "k": 10.0,
    "tau": 0.6,
    "min": 2.0,
    "max": 20.0,
    "display_cap": 10.0
}


def decode_event_from_obs(obs: np.ndarray) -> Event:
    """
    Decode event from observation vector.
    Event one-hot is at indices 23-28.
    """
    event_onehot = obs[23:29]
    event_types = [EVENT_NONE, EVENT_GE10, EVENT_LE7, EVENT_EVEN, EVENT_ODD, EVENT_REMAP_VALUE]
    
    idx = np.argmax(event_onehot)
    if event_onehot[idx] > 0.5:
        event_type = event_types[idx]
    else:
        event_type = EVENT_NONE
    
    # For remap_value, we don't know the params from obs alone
    # This is a limitation - remap events are rare in default config
    return Event(type=event_type)


def decode_hints_from_obs(obs: np.ndarray) -> List[int]:
    """
    Decode revealed hints from observation vector.
    Hints are at indices 10-22 as count vector for ranks 2..14.
    """
    hint_counts = obs[10:23]
    hints = []
    for i, count in enumerate(hint_counts):
        rank = i + 2  # ranks 2..14
        for _ in range(int(round(count))):
            hints.append(rank)
    return hints


def compute_fair_value_from_obs(obs: np.ndarray) -> Tuple[float, float]:
    """
    Compute the Bayesian fair value (posterior mean and std) using only
    observable information from the observation vector.
    
    Returns:
        mu: Expected sum E[S | hints, event]
        sigma: Std dev of sum
    """
    # Decode event
    event = decode_event_from_obs(obs)
    
    # Decode hints
    hints = decode_hints_from_obs(obs)
    
    # Apply event to get filtered deck and value map
    filtered_deck, value_map = apply_event(FULL_DECK, event)
    
    # Compute posterior using only observable information
    mu, sigma = posterior_mean_var(hints, filtered_deck, value_map)
    
    return mu, sigma


def compute_expected_overflow(
    order_size: float,
    D_disp: float,
    L_cap: float,
    liq_cfg: Dict[str, Any]
) -> float:
    """
    Compute E[(order_size - L_true)+ | D_disp] where D_disp = min(L_true, L_cap).
    
    Two cases:
    1. D_disp < L_cap: Then L_true = D_disp exactly (no uncertainty)
       E[(order_size - L_true)+] = max(0, order_size - D_disp)
    
    2. D_disp = L_cap: Then L_true >= L_cap (truncated distribution)
       Need to integrate over the conditional distribution of L_true | L_true >= L_cap
    """
    tau = liq_cfg.get("tau", 0.6)
    L_min = liq_cfg.get("min", 2.0)
    L_max = liq_cfg.get("max", 20.0)
    
    # Case 1: We know true depth exactly
    if D_disp < L_cap - 1e-6:
        L_true_known = D_disp
        return max(0.0, order_size - L_true_known)
    
    # Case 2: Truncated distribution - L_true >= L_cap
    # We need to compute E[(order_size - L)+ | L >= L_cap]
    
    # The lognormal parameters depend on market conditions (sigma_S, spread)
    # Without direct access, we use a reasonable approximation based on L_cap
    # being a reasonable estimate of L_bar (the nominal depth scale)
    
    # Approximate: assume L_bar ≈ L_cap (market maker shows cap when depth is high)
    L_bar = L_cap * 1.2  # slightly above cap as reasonable prior
    mu_ln = np.log(L_bar) - 0.5 * (tau ** 2)
    
    # For L ~ LogNormal(mu_ln, tau) truncated to [L_cap, L_max]
    # E[(order_size - L)+ | L >= L_cap] = E[(order_size - L) * I(L < order_size) | L >= L_cap]
    
    if order_size <= L_cap:
        # If order_size <= L_cap and L >= L_cap, then (order_size - L)+ = 0 always
        return 0.0
    
    if order_size >= L_max:
        # E[(order_size - L)+ | L in [L_cap, L_max]] = order_size - E[L | L in [L_cap, L_max]]
        # Use numerical integration
        pass
    
    # Numerical integration for the general case
    # P(L >= L_cap) under the uncapped lognormal
    dist = stats.lognorm(s=tau, scale=np.exp(mu_ln))
    
    # CDF values for truncation
    F_cap = dist.cdf(L_cap)
    F_max = dist.cdf(L_max)
    
    # Truncation normalization constant
    norm_const = F_max - F_cap
    if norm_const < 1e-10:
        # Almost all mass is below L_cap, so expect L ≈ L_cap
        return max(0.0, order_size - L_cap)
    
    # Compute E[(order_size - L)+ | L in [L_cap, L_max]]
    # = integral from L_cap to min(order_size, L_max) of (order_size - L) * f(L) dL / norm_const
    
    upper_bound = min(order_size, L_max)
    if upper_bound <= L_cap:
        return 0.0
    
    def integrand(L):
        return (order_size - L) * dist.pdf(L)
    
    result, _ = quad(integrand, L_cap, upper_bound, limit=50)
    
    return result / norm_const


def compute_expected_buy_edge(
    size: int,
    mu: float,
    Y: float,  # ask price
    D_disp_ask: float,
    alpha: float,
    enable_impact: bool,
    liq_cfg: Dict[str, Any]
) -> float:
    """
    Compute expected edge for a buy order of given size.
    
    Without impact: edge = size * (mu - Y)
    With impact: edge = size * (mu - E[p_exec_buy])
                      = size * (mu - Y - alpha * E[(size - L_true)+])
    """
    if not enable_impact:
        return size * (mu - Y)
    
    L_cap = liq_cfg.get("display_cap", 10.0)
    expected_overflow = compute_expected_overflow(size, D_disp_ask, L_cap, liq_cfg)
    expected_exec_price = Y + alpha * expected_overflow
    
    return size * (mu - expected_exec_price)


def compute_expected_sell_edge(
    size: int,
    mu: float,
    X: float,  # bid price
    D_disp_bid: float,
    alpha: float,
    enable_impact: bool,
    liq_cfg: Dict[str, Any]
) -> float:
    """
    Compute expected edge for a sell order of given size.
    
    Without impact: edge = size * (X - mu)
    With impact: edge = size * (E[p_exec_sell] - mu)
                      = size * (X - alpha * E[(size - L_true)+] - mu)
    """
    if not enable_impact:
        return size * (X - mu)
    
    L_cap = liq_cfg.get("display_cap", 10.0)
    expected_overflow = compute_expected_overflow(size, D_disp_bid, L_cap, liq_cfg)
    expected_exec_price = X - alpha * expected_overflow
    
    return size * (expected_exec_price - mu)


def _get_fallback_action(mask: np.ndarray) -> int:
    """
    Get a valid fallback action when preferred action is not available.
    Prefers Pass, then smallest valid trade.
    """
    # Prefer Pass
    if mask[0]:
        return 0
    
    # Find smallest valid buy
    for i in range(1, 11):
        if mask[i]:
            return i
    
    # Find smallest valid sell
    for i in range(11, 21):
        if mask[i]:
            return i
    
    # Should never reach here if mask has any valid action
    return 0


def act_ev_realistic(
    obs: np.ndarray,
    mask: np.ndarray,
    enable_impact: bool = False,
    alpha: float = 0.15,
    liq_cfg: Optional[Dict[str, Any]] = None
) -> int:
    """
    EV Realistic policy action selection.
    
    Uses only observable information to compute fair value and select
    the action that maximizes expected edge.
    
    Args:
        obs: Observation vector (35-dim)
        mask: Valid action mask (21-dim)
        enable_impact: Whether market impact is enabled
        alpha: Impact coefficient
        liq_cfg: Liquidity configuration
    
    Returns:
        Action index (0=Pass, 1-10=Buy, 11-20=Sell)
    """
    if liq_cfg is None:
        liq_cfg = DEFAULT_LIQUIDITY_CFG.copy()
    
    # Extract observable quantities from obs
    X = obs[0]  # bid
    Y = obs[1]  # ask
    D_disp_bid = obs[4]
    D_disp_ask = obs[5]
    
    # Check if impact is enabled from obs flags (index 32)
    # But we use the passed parameter for flexibility
    
    # Compute fair value using only observable information
    mu, sigma = compute_fair_value_from_obs(obs)
    
    # Simple case: Impact OFF
    if not enable_impact:
        edge_buy = mu - Y
        edge_sell = X - mu
        
        if edge_buy > 0 and edge_buy >= edge_sell:
            # Buy - find max valid size
            for i in range(10, 0, -1):
                if mask[i]:
                    return i
                    
        elif edge_sell > 0 and edge_sell > edge_buy:
            # Sell - find max valid size
            for i in range(10, 0, -1):
                idx = 10 + i
                if mask[idx]:
                    return idx
        
        return _get_fallback_action(mask)  # Pass if valid, else fallback
    
    # Impact ON: Find size that maximizes expected edge
    best_action = 0
    best_edge = 0.0
    
    # Evaluate buy actions
    for i in range(1, 11):
        if mask[i]:
            edge = compute_expected_buy_edge(
                i, mu, Y, D_disp_ask, alpha, True, liq_cfg
            )
            if edge > best_edge:
                best_edge = edge
                best_action = i
    
    # Evaluate sell actions
    for i in range(1, 11):
        idx = 10 + i
        if mask[idx]:
            edge = compute_expected_sell_edge(
                i, mu, X, D_disp_bid, alpha, True, liq_cfg
            )
            if edge > best_edge:
                best_edge = edge
                best_action = idx
    
    # If no positive edge found, ensure we return a valid action
    if best_action == 0 and not mask[0]:
        return _get_fallback_action(mask)
    
    return best_action


class EVRealisticAgent:
    """
    EV Realistic baseline agent.
    
    Computes fair value using Bayesian inference from observable information
    only (hints + event), without access to privileged information like
    true_sum or true_depths.
    """
    
    def __init__(
        self,
        alpha: float = 0.15,
        liq_cfg: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            alpha: Market impact coefficient
            liq_cfg: Liquidity configuration dict
        """
        self.alpha = alpha
        self.liq_cfg = liq_cfg or DEFAULT_LIQUIDITY_CFG.copy()
    
    def act(
        self,
        obs: np.ndarray,
        mask: np.ndarray,
        info: Optional[Dict[str, Any]] = None,
        eval_mode: bool = True
    ) -> int:
        """
        Select action based on observable information only.
        
        Note: The info dict is accepted for API compatibility but NOT used
        for decision making (no peeking at mu, true_sum, etc.).
        
        Args:
            obs: Observation vector
            mask: Valid action mask
            info: Info dict (IGNORED - for API compatibility only)
            eval_mode: Evaluation mode flag
        
        Returns:
            Action index
        """
        # Determine if impact is enabled from observation flags
        enable_impact = obs[32] > 0.5 if len(obs) > 32 else False
        
        return act_ev_realistic(
            obs, mask,
            enable_impact=enable_impact,
            alpha=self.alpha,
            liq_cfg=self.liq_cfg
        )
    
    def get_fair_value(self, obs: np.ndarray) -> Tuple[float, float]:
        """
        Get the computed fair value and uncertainty from observation.
        Useful for debugging/analysis.
        
        Returns:
            (mu, sigma): Fair value mean and std
        """
        return compute_fair_value_from_obs(obs)


# Convenience function for evaluation scripts
def create_ev_realistic_agent(
    alpha: float = 0.15,
    liq_cfg: Optional[Dict[str, Any]] = None
) -> EVRealisticAgent:
    """Create an EV Realistic agent with specified parameters."""
    return EVRealisticAgent(alpha=alpha, liq_cfg=liq_cfg)

