import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple, Optional, List

from mmrl.env.cards import make_deck, draw_cards, calculate_sum
from mmrl.env.events import sample_event, apply_event, Event
from mmrl.env.quotes import posterior_mean_var, make_quote
from mmrl.env.liquidity import draw_true_depth, displayed_depth, exec_price_buy, exec_price_sell
from mmrl.env.execution import get_action_mask, ACTION_PASS
from mmrl.env.spaces import ACTION_SPACE, get_obs_shape, build_obs
from mmrl.env.logging_hooks import make_log_entry

class TwoPlayerCardEnv(gym.Env):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        
        self.action_space = ACTION_SPACE
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=get_obs_shape(), dtype=np.float32
        )
        
        self.rng = np.random.RandomState(42)
        self.t = 0
        self.T = self._get_cfg("episode_length", 10)
        self.W0 = self._get_cfg("W0", 500.0)
        
        # Agents state
        self.W_a = self.W0
        self.W_b = self.W0
        
        self.current_event: Optional[Event] = None
        
        # Last actions (side, size)
        self.last_act_a = {"side": 0.0, "size": 0.0}
        self.last_act_b = {"side": 0.0, "size": 0.0}
        
        # Metrics
        self.metrics_a = {}
        self.metrics_b = {}
        self.true_sum = 0
        self.mu = 0.0
        self.sigma = 0.0
        
    def _get_cfg(self, key, default=None):
        keys = key.split(".")
        val = self.cfg
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k)
            else:
                val = getattr(val, k, None)
            if val is None:
                return default
        return val

    def reset(
        self, 
        *, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.RandomState(seed)
            
        self.t = 0
        self.W_a = self.W0
        self.W_b = self.W0
        self.last_act_a = {"side": 0.0, "size": 0.0}
        self.last_act_b = {"side": 0.0, "size": 0.0}
        self.metrics_a = {}
        self.metrics_b = {}
        self.current_event = None
        
        return self._start_round()

    def _sample_hints(self, hidden_cards: List[int]) -> List[int]:
        n_hints = self._get_cfg("hints.count", self.rng.randint(0, 4))
        n_hints = min(n_hints, 3)
        indices = self.rng.choice(3, size=n_hints, replace=False)
        return [hidden_cards[i] for i in indices]

    def _start_round(self) -> Tuple[Tuple[np.ndarray, np.ndarray], Dict[str, Any]]:
        # 1. Event
        self.current_event = sample_event(self.rng, self.cfg, last_event=self.current_event)
        
        # 2. Cards
        full_deck = make_deck()
        filtered_deck, value_map = apply_event(full_deck, self.current_event)
        self.hidden_cards, _ = draw_cards(filtered_deck, 3, self.rng)
        self.true_sum = sum([value_map[c] for c in self.hidden_cards])
        
        # 3. Hints
        self.hints_shared = self._sample_hints(self.hidden_cards)
        self.hints_a = self.hints_shared
        self.hints_b = self.hints_shared
        
        # 4. Posterior & Quote
        self.mu, self.sigma = posterior_mean_var(self.hints_shared, filtered_deck, value_map)
        self.quote = make_quote(self.mu, self.sigma, self.cfg, self.rng)
        
        # 5. Liquidity
        l_bid = draw_true_depth(self.sigma, self.quote.spread, self.cfg, self.rng)
        l_ask = draw_true_depth(self.sigma, self.quote.spread, self.cfg, self.rng)
        self.true_depths = (l_bid, l_ask)
        
        d_bid = displayed_depth(l_bid, self.cfg)
        d_ask = displayed_depth(l_ask, self.cfg)
        self.disp_depths = (d_bid, d_ask)
        
        # 6. Build Obs
        flags = {
            "enable_events": self._get_cfg("flags.enable_events", False),
            "enable_impact": self._get_cfg("flags.enable_impact", False)
        }
        
        obs_a = build_obs(
            quote=self.quote, depths=self.disp_depths, metrics=self.metrics_a,
            hints=self.hints_a, event=self.current_event,
            state={"W": self.W_a, "W0": self.W0, "t": self.t, "T": self.T},
            flags=flags, opponent_last=self.last_act_b
        )
        
        obs_b = build_obs(
            quote=self.quote, depths=self.disp_depths, metrics=self.metrics_b,
            hints=self.hints_b, event=self.current_event,
            state={"W": self.W_b, "W0": self.W0, "t": self.t, "T": self.T},
            flags=flags, opponent_last=self.last_act_a
        )
        
        info = {
            "mu": self.mu,
            "sigma": self.sigma,
            "true_sum": self.true_sum,
            "mask_a": get_action_mask(self.W_a, self.quote.bid, self.quote.ask, self.current_event),
            "mask_b": get_action_mask(self.W_b, self.quote.bid, self.quote.ask, self.current_event)
        }
        
        return (obs_a, obs_b), info
    
    def _get_mask(self, agent_idx: int) -> np.ndarray:
        """Helper to get current action mask for an agent (0 or 1)."""
        if agent_idx == 0:
            return get_action_mask(self.W_a, self.quote.bid, self.quote.ask, self.current_event)
        elif agent_idx == 1:
            return get_action_mask(self.W_b, self.quote.bid, self.quote.ask, self.current_event)
        else:
            raise ValueError("Agent index must be 0 or 1")

    def step(
        self, 
        actions: Tuple[int, int]
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[float, float], bool, bool, Dict[str, Any]]:
        
        act_a, act_b = actions
        
        # Mask validation
        mask_a = get_action_mask(self.W_a, self.quote.bid, self.quote.ask, self.current_event)
        if not mask_a[act_a]:
            act_a = 0
            
        mask_b = get_action_mask(self.W_b, self.quote.bid, self.quote.ask, self.current_event)
        if not mask_b[act_b]:
            act_b = 0
            
        # Decode actions
        # 0=Pass, 1..10=Buy, 11..20=Sell
        def decode(a):
            if a == 0: return 0, 0.0
            if 1 <= a <= 10: return 1, float(a)
            if 11 <= a <= 20: return -1, float(a - 10)
            return 0, 0.0
            
        side_a, size_a = decode(act_a)
        side_b, size_b = decode(act_b)
        
        # Aggregate
        q_buy = 0.0
        q_sell = 0.0
        
        if side_a == 1: q_buy += size_a
        elif side_a == -1: q_sell += size_a
        
        if side_b == 1: q_buy += size_b
        elif side_b == -1: q_sell += size_b
        
        # Execute
        enable_impact = self._get_cfg("flags.enable_impact", False)
        alpha = self._get_cfg("alpha", 0.3)
        
        p_exec_buy = exec_price_buy(self.quote.ask, q_buy, self.true_depths[1], alpha, enable_impact)
        p_exec_sell = exec_price_sell(self.quote.bid, q_sell, self.true_depths[0], alpha, enable_impact)
        
        # Rewards
        def calc_reward(side, size):
            if side == 0: return 0.0
            if side == 1: return size * (self.true_sum - p_exec_buy)
            if side == -1: return size * (p_exec_sell - self.true_sum)
            return 0.0
            
        r_a = calc_reward(side_a, size_a)
        r_b = calc_reward(side_b, size_b)
        
        # Update State
        self.W_a += r_a
        self.W_b += r_b
        
        self.last_act_a = {"side": float(side_a), "size": size_a}
        self.last_act_b = {"side": float(side_b), "size": size_b}
        
        # Metrics
        def calc_metrics(side, price):
            slip = 0.0
            if side == 1: slip = price - self.quote.ask
            elif side == -1: slip = self.quote.bid - price
            
            return {
                "slippage_bid": slip if side == -1 else 0.0,
                "slippage_ask": slip if side == 1 else 0.0,
                "fill_ratio_bid": 1.0 if side == -1 else 0.0,
                "fill_ratio_ask": 1.0 if side == 1 else 0.0
            }
            
        self.metrics_a = calc_metrics(side_a, p_exec_buy if side_a == 1 else p_exec_sell)
        self.metrics_b = calc_metrics(side_b, p_exec_buy if side_b == 1 else p_exec_sell)
        
        # Time
        self.t += 1
        
        terminated = False
        truncated = False
        if self.t >= self.T:
            terminated = True
            
        stop_out_frac = self._get_cfg("stop_out", 0.2)
        if self.W_a < stop_out_frac * self.W0 or self.W_b < stop_out_frac * self.W0:
            terminated = True
            
        # Info
        # Include masks so that downstream agents (e.g. IPPO) can always
        # access valid-action masks, even on terminal transitions.
        step_info = {
            "exec_price_buy": p_exec_buy,
            "exec_price_sell": p_exec_sell,
            "q_buy_total": q_buy,
            "q_sell_total": q_sell,
            "true_sum": self.true_sum,
            "mask_a": mask_a,
            "mask_b": mask_b,
        }
        
        # Logs
        state_info = {
            "t": self.t - 1,
            "mu": self.mu,
            "sigma": self.sigma,
            "event": self.current_event.to_dict(),
            "true_depths": self.true_depths
        }
        
        state_info_a = state_info.copy()
        state_info_a["W"] = self.W_a - r_a
        log_a = make_log_entry(
            {"reward": r_a, "exec_price": p_exec_buy if side_a==1 else p_exec_sell, "slippage": self.metrics_a["slippage_bid"]+self.metrics_a["slippage_ask"], "true_sum": self.true_sum},
            None, act_a, mask_a, state_info_a
        )
        
        state_info_b = state_info.copy()
        state_info_b["W"] = self.W_b - r_b
        log_b = make_log_entry(
            {"reward": r_b, "exec_price": p_exec_buy if side_b==1 else p_exec_sell, "slippage": self.metrics_b["slippage_bid"]+self.metrics_b["slippage_ask"], "true_sum": self.true_sum},
            None, act_b, mask_b, state_info_b
        )
        
        if not terminated:
            (obs_a, obs_b), info_next = self._start_round()
            info_next.update(step_info)
            info_next["log_a"] = log_a
            info_next["log_b"] = log_b
            return (obs_a, obs_b), (r_a, r_b), terminated, truncated, info_next
        else:
            # Terminal obs
            flags = {
                "enable_events": self._get_cfg("flags.enable_events", False),
                "enable_impact": self._get_cfg("flags.enable_impact", False)
            }
            obs_a = build_obs(self.quote, self.disp_depths, self.metrics_a, self.hints_a, self.current_event, {"W": self.W_a, "W0": self.W0, "t": self.t, "T": self.T}, flags, self.last_act_b)
            obs_b = build_obs(self.quote, self.disp_depths, self.metrics_b, self.hints_b, self.current_event, {"W": self.W_b, "W0": self.W0, "t": self.t, "T": self.T}, flags, self.last_act_a)
            
            step_info["log_a"] = log_a
            step_info["log_b"] = log_b
            
            return (obs_a, obs_b), (r_a, r_b), terminated, truncated, step_info
