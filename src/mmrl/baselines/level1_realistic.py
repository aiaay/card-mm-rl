"""
Level-1 Realistic Baseline (No Privileged Information)

A strategic baseline that models opponent behavior and accounts for crowding,
using ONLY observable information:
- Quote (bid/ask)
- Event (one-hot encoded in obs)
- Hints (revealed cards)
- Displayed depth (NOT true depth)
- Opponent's last action (from obs)

This is the "realistic" version of Level1Policy that does NOT peek at:
- info["mu"] (true fair value)
- info["true_depths"] (true liquidity)

Instead, it:
1. Computes fair value using Bayesian inference (like EV Realistic)
2. Tracks opponent history to estimate crowding
3. Reasons about expected impact with uncertain liquidity
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple
from collections import deque
from scipy import stats
from scipy.integrate import quad

from mmrl.baselines.ev_realistic import (
    compute_fair_value_from_obs,
    compute_expected_overflow,
    DEFAULT_LIQUIDITY_CFG
)


class Level1RealisticPolicy:
    """
    Level-1 Crowding-Aware Policy without privileged information.
    
    Combines:
    - Bayesian fair value estimation from observable data
    - Opponent modeling through action history tracking
    - Expected impact calculation with uncertain liquidity
    
    This represents a sophisticated baseline that a real trader
    could implement without access to hidden market information.
    """
    
    def __init__(
        self,
        history_len: int = 10,
        alpha: float = 0.15,
        liq_cfg: Optional[Dict[str, Any]] = None,
        risk_aversion: float = 0.5
    ):
        """
        Args:
            history_len: Number of opponent actions to track
            alpha: Market impact coefficient
            liq_cfg: Liquidity configuration
            risk_aversion: How much to discount edge for uncertainty (0-1)
                          0 = risk neutral, 1 = very risk averse
        """
        self.history_len = history_len
        self.alpha = alpha
        self.liq_cfg = liq_cfg or DEFAULT_LIQUIDITY_CFG.copy()
        self.risk_aversion = risk_aversion
        
        # History of opponent actions: list of (side, size)
        self.opp_history: deque = deque(maxlen=history_len)
        
    def reset(self):
        """Clear opponent history (call at episode start if needed)."""
        self.opp_history.clear()
        
    def update(self, opp_side: int, opp_size: float):
        """
        Update belief about opponent based on last round.
        
        Args:
            opp_side: -1 (sell), 0 (pass), 1 (buy)
            opp_size: Size of opponent's order (0 if pass)
        """
        self.opp_history.append((opp_side, opp_size))
        
    def _estimate_opp_demand(self, my_side: int) -> Tuple[float, float]:
        """
        Estimate expected opponent demand on the same side.
        
        Returns:
            (expected_demand, demand_uncertainty): Mean and std of opponent demand
        """
        if len(self.opp_history) == 0:
            # Prior: assume opponent may trade with 50% prob, avg size 3
            return 1.5, 2.0
            
        # Frequency of opponent being on the same side
        same_side_trades = [(s, z) for s, z in self.opp_history if s == my_side]
        count_same = len(same_side_trades)
        prob = count_same / len(self.opp_history)
        
        # Size statistics when on that side
        if count_same > 0:
            sizes = [z for s, z in same_side_trades]
            avg_size = np.mean(sizes)
            std_size = np.std(sizes) if len(sizes) > 1 else 2.0
        else:
            avg_size = 5.0  # Default guess
            std_size = 2.0
        
        # Expected opponent demand
        expected_demand = prob * avg_size
        
        # Uncertainty in demand estimate
        # Combines uncertainty in prob and size
        demand_uncertainty = np.sqrt(
            (prob * std_size) ** 2 + 
            (avg_size * np.sqrt(prob * (1 - prob) / max(1, len(self.opp_history)))) ** 2
        )
        
        return expected_demand, demand_uncertainty
    
    def _compute_expected_total_overflow(
        self,
        my_size: float,
        opp_demand: float,
        opp_uncertainty: float,
        D_disp: float,
        enable_impact: bool
    ) -> float:
        """
        Compute expected overflow considering both liquidity and opponent uncertainty.
        
        Total demand = my_size + opponent_demand
        Overflow = (total_demand - L_true)+
        
        We need E[(my_size + opp - L)+] where both opp and L are uncertain.
        """
        if not enable_impact:
            return 0.0
        
        L_cap = self.liq_cfg.get("display_cap", 10.0)
        
        # Case 1: Known depth (D_disp < L_cap)
        if D_disp < L_cap - 1e-6:
            L_true = D_disp
            # Only opponent demand is uncertain
            # E[(my_size + opp - L)+] ≈ max(0, my_size + E[opp] - L) 
            # plus adjustment for uncertainty
            base_overflow = max(0.0, my_size + opp_demand - L_true)
            
            # Add risk-averse adjustment: consider upside risk of opponent demand
            risk_adj = self.risk_aversion * opp_uncertainty * 0.5
            
            return base_overflow + risk_adj
        
        # Case 2: Uncertain depth (D_disp = L_cap)
        # Both L_true and opponent demand are uncertain
        # Approximate: treat opponent demand as expected value, compute E[(size + opp - L)+]
        total_demand = my_size + opp_demand
        base_overflow = compute_expected_overflow(
            total_demand, D_disp, L_cap, self.liq_cfg
        )
        
        # Risk adjustment for opponent uncertainty
        # If opponent trades more than expected, overflow increases
        risk_adj = self.risk_aversion * opp_uncertainty * 0.3
        
        return base_overflow + risk_adj
    
    def _compute_edge(
        self,
        side: int,  # 1 for buy, -1 for sell
        size: int,
        mu: float,
        sigma: float,
        X: float,  # bid
        Y: float,  # ask
        D_disp: float,  # displayed depth on relevant side
        opp_demand: float,
        opp_uncertainty: float,
        enable_impact: bool
    ) -> float:
        """
        Compute risk-adjusted expected edge for a trade.
        
        Args:
            side: 1 for buy, -1 for sell
            size: Order size
            mu: Fair value estimate
            sigma: Uncertainty in fair value
            X, Y: Bid and ask prices
            D_disp: Displayed depth
            opp_demand: Expected opponent demand on same side
            opp_uncertainty: Uncertainty in opponent demand
            enable_impact: Whether market impact is active
        
        Returns:
            Risk-adjusted expected edge
        """
        expected_overflow = self._compute_expected_total_overflow(
            float(size), opp_demand, opp_uncertainty, D_disp, enable_impact
        )
        
        if side == 1:  # Buy
            expected_price = Y + self.alpha * expected_overflow
            raw_edge = size * (mu - expected_price)
        else:  # Sell
            expected_price = X - self.alpha * expected_overflow
            raw_edge = size * (expected_price - mu)
        
        # Risk adjustment for fair value uncertainty
        # Reduce edge if sigma is high (less confident in direction)
        risk_penalty = self.risk_aversion * sigma * size * 0.1
        
        return raw_edge - risk_penalty
    
    def _extract_opponent_action_from_obs(self, obs: np.ndarray) -> Tuple[int, float]:
        """
        Extract opponent's last action from observation vector.
        Opponent info is at indices 33-34: (side, size)
        """
        if len(obs) >= 35:
            opp_side = int(round(obs[33]))  # -1, 0, or 1
            opp_size = obs[34]
            return opp_side, opp_size
        return 0, 0.0
    
    def _get_fallback_action(self, mask: np.ndarray) -> int:
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
    
    def act(
        self,
        obs: np.ndarray,
        mask: np.ndarray,
        info: Optional[Dict[str, Any]] = None,
        eval_mode: bool = True
    ) -> int:
        """
        Select action using Level-1 reasoning without privileged information.
        
        Args:
            obs: Observation vector
            mask: Valid action mask
            info: Info dict (IGNORED - for API compatibility only)
            eval_mode: Evaluation mode flag
        
        Returns:
            Action index (0=Pass, 1-10=Buy, 11-20=Sell)
        """
        # Extract observable quantities
        X = obs[0]  # bid
        Y = obs[1]  # ask
        D_disp_bid = obs[4]
        D_disp_ask = obs[5]
        
        # Check if impact is enabled from observation
        enable_impact = obs[32] > 0.5 if len(obs) > 32 else False
        
        # Compute fair value from observable information only
        mu, sigma = compute_fair_value_from_obs(obs)
        
        # Determine direction based on edge
        edge_buy = mu - Y
        edge_sell = X - mu
        
        if edge_buy <= 0 and edge_sell <= 0:
            # No edge - prefer Pass if valid, otherwise fallback
            return self._get_fallback_action(mask)
        
        # Determine primary direction
        if edge_buy > edge_sell:
            primary_side = 1  # Buy
            secondary_side = -1
        else:
            primary_side = -1  # Sell
            secondary_side = 1
        
        # Update opponent history from observation (if visible)
        opp_side, opp_size = self._extract_opponent_action_from_obs(obs)
        if opp_side != 0 or opp_size > 0:
            # Only update if there's a meaningful signal
            # Note: In two-player env, opponent action is visible
            pass  # History is updated externally via update() in eval loop
        
        # Estimate opponent crowding
        opp_demand_primary, opp_unc_primary = self._estimate_opp_demand(primary_side)
        
        # Find best action considering crowding and impact
        best_action = 0
        best_edge = 0.0
        
        # Evaluate primary direction first
        if primary_side == 1:
            # Buy side
            for i in range(10, 0, -1):
                if mask[i]:
                    edge = self._compute_edge(
                        side=1, size=i, mu=mu, sigma=sigma,
                        X=X, Y=Y, D_disp=D_disp_ask,
                        opp_demand=opp_demand_primary,
                        opp_uncertainty=opp_unc_primary,
                        enable_impact=enable_impact
                    )
                    if edge > best_edge:
                        best_edge = edge
                        best_action = i
        else:
            # Sell side
            for i in range(10, 0, -1):
                idx = 10 + i
                if mask[idx]:
                    edge = self._compute_edge(
                        side=-1, size=i, mu=mu, sigma=sigma,
                        X=X, Y=Y, D_disp=D_disp_bid,
                        opp_demand=opp_demand_primary,
                        opp_uncertainty=opp_unc_primary,
                        enable_impact=enable_impact
                    )
                    if edge > best_edge:
                        best_edge = edge
                        best_action = idx
        
        # Also check secondary direction if primary has negative edge after crowding
        if best_edge <= 0:
            opp_demand_secondary, opp_unc_secondary = self._estimate_opp_demand(secondary_side)
            
            if secondary_side == 1:
                for i in range(10, 0, -1):
                    if mask[i]:
                        edge = self._compute_edge(
                            side=1, size=i, mu=mu, sigma=sigma,
                            X=X, Y=Y, D_disp=D_disp_ask,
                            opp_demand=opp_demand_secondary,
                            opp_uncertainty=opp_unc_secondary,
                            enable_impact=enable_impact
                        )
                        if edge > best_edge:
                            best_edge = edge
                            best_action = i
            else:
                for i in range(10, 0, -1):
                    idx = 10 + i
                    if mask[idx]:
                        edge = self._compute_edge(
                            side=-1, size=i, mu=mu, sigma=sigma,
                            X=X, Y=Y, D_disp=D_disp_bid,
                            opp_demand=opp_demand_secondary,
                            opp_uncertainty=opp_unc_secondary,
                            enable_impact=enable_impact
                        )
                        if edge > best_edge:
                            best_edge = edge
                            best_action = idx
        
        # If no positive edge found, ensure we return a valid action
        if best_action == 0 and not mask[0]:
            return self._get_fallback_action(mask)
        
        return best_action
    
    def get_fair_value(self, obs: np.ndarray) -> Tuple[float, float]:
        """
        Get the computed fair value from observation.
        Useful for debugging/analysis.
        
        Returns:
            (mu, sigma): Fair value mean and std
        """
        return compute_fair_value_from_obs(obs)
    
    def get_opponent_estimate(self, side: int) -> Tuple[float, float]:
        """
        Get current estimate of opponent demand on a given side.
        
        Args:
            side: 1 for buy, -1 for sell
        
        Returns:
            (expected_demand, uncertainty)
        """
        return self._estimate_opp_demand(side)


# Convenience function
def create_level1_realistic_agent(
    history_len: int = 10,
    alpha: float = 0.15,
    risk_aversion: float = 0.5
) -> Level1RealisticPolicy:
    """Create a Level-1 Realistic agent with specified parameters."""
    return Level1RealisticPolicy(
        history_len=history_len,
        alpha=alpha,
        risk_aversion=risk_aversion
    )

