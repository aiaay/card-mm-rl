import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple, Optional

from mmrl.env.cards import make_deck, draw_cards, calculate_sum
from mmrl.env.events import sample_event, apply_event, Event
from mmrl.env.quotes import posterior_mean_var, make_quote
from mmrl.env.liquidity import draw_true_depth, displayed_depth, exec_price_buy, exec_price_sell
from mmrl.env.execution import get_action_mask, ACTION_PASS
from mmrl.env.spaces import ACTION_SPACE, get_obs_shape, build_obs
from mmrl.env.logging_hooks import make_log_entry

class SingleCardEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        
        # Spaces
        self.action_space = ACTION_SPACE
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=get_obs_shape(), dtype=np.float32
        )
        
        # State
        self.rng = np.random.RandomState(42)
        self.t = 0
        self.T = self._get_cfg("episode_length", 10)
        self.W0 = self._get_cfg("W0", 500.0)
        self.W = self.W0
        self.current_event: Optional[Event] = None
        
        # Per-round state
        self.hidden_cards = []
        self.hints = []
        self.quote = None
        self.true_depths = (0.0, 0.0) # bid, ask
        self.disp_depths = (0.0, 0.0)
        self.metrics = {} # slippage, fills from *last* step
        self.true_sum = 0
        self.mu = 0.0
        self.sigma = 0.0
        
        # Pass tracking
        self.consecutive_passes = 0
        self.max_consecutive_passes = self._get_cfg("max_consecutive_passes", 3)
        
    def _get_cfg(self, key, default=None):
        # Helper to access nested dict config
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
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.RandomState(seed)
            
        self.t = 0
        self.W = self.W0
        self.metrics = {} # Clear metrics
        self.current_event = None
        self.consecutive_passes = 0
        self.max_consecutive_passes = self._get_cfg("max_consecutive_passes", 3)
        
        return self._start_round()

    def _get_current_mask(self):
        mask = get_action_mask(self.W, self.quote.bid, self.quote.ask, self.current_event)
        
        # Enforce consecutive pass limit
        if self.consecutive_passes >= self.max_consecutive_passes:
            # Check if any other action is possible
            if np.sum(mask[1:]) > 0:
                mask[0] = False # Forbid pass
                
        return mask

    def _start_round(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Logic to start a new round (t):
        1. Sample Event
        2. Draw Cards
        3. Sample Hints
        4. Compute Posterior & Quote
        5. Draw Liquidity
        6. Build Obs
        """
        # 1. Event
        self.current_event = sample_event(self.rng, self.cfg, last_event=self.current_event)
        
        # 2. Cards
        full_deck = make_deck()
        filtered_deck, value_map = apply_event(full_deck, self.current_event)
        
        # Draw 3 cards
        self.hidden_cards, _ = draw_cards(filtered_deck, 3, self.rng)
        self.true_sum = sum([value_map[c] for c in self.hidden_cards])
        
        # 3. Hints
        hints_cfg = self._get_cfg("hints", {})
        forced_hint_count = None
        if isinstance(hints_cfg, dict):
            forced_hint_count = hints_cfg.get("count")
        if forced_hint_count is not None:
            n_hints = int(forced_hint_count)
        else:
            probs = None
            if isinstance(hints_cfg, dict):
                probs = hints_cfg.get("probabilities")
            if probs is not None:
                probs = np.array(probs, dtype=float)
                if probs.shape[0] != 4 or probs.sum() <= 0:
                    probs = None
            if probs is None:
                probs = np.array([0.35, 0.30, 0.20, 0.15])
            probs = probs / probs.sum()
            n_hints = int(self.rng.choice([0, 1, 2, 3], p=probs))
        n_hints = max(0, min(n_hints, 3))
        
        hint_indices = self.rng.choice(3, size=n_hints, replace=False) if n_hints > 0 else []
        self.hints = [self.hidden_cards[i] for i in hint_indices]
        
        # 4. Posterior & Quote
        self.mu, self.sigma = posterior_mean_var(self.hints, filtered_deck, value_map)
        self.quote = make_quote(self.mu, self.sigma, self.cfg, self.rng)
        
        # 5. Liquidity
        l_bid = draw_true_depth(self.sigma, self.quote.spread, self.cfg, self.rng)
        l_ask = draw_true_depth(self.sigma, self.quote.spread, self.cfg, self.rng)
        self.true_depths = (l_bid, l_ask)
        
        d_bid = displayed_depth(l_bid, self.cfg)
        d_ask = displayed_depth(l_ask, self.cfg)
        self.disp_depths = (d_bid, d_ask)
        
        # 6. Build Obs
        obs = build_obs(
            quote=self.quote,
            depths=self.disp_depths,
            metrics=self.metrics,
            hints=self.hints,
            event=self.current_event,
            state={"W": self.W, "W0": self.W0, "t": self.t, "T": self.T},
            flags={
                "enable_events": self._get_cfg("flags.enable_events", False),
                "enable_impact": self._get_cfg("flags.enable_impact", False)
            }
        )
        
        # Info
        info = {
            "mu": self.mu,
            "sigma": self.sigma,
            "event": self.current_event.to_dict(),
            "hidden_cards": self.hidden_cards,
            "true_sum": self.true_sum,
            "true_depths": self.true_depths,
            "mask": self._get_current_mask()
        }
        
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Validate action
        # 0: Pass, 1..10: Buy, 11..20: Sell
        
        # Retrieve Mask to check validity
        mask = self._get_current_mask()
        
        if not mask[action]:
            # Force valid action
            # If tried to pass but forbidden, force action 0? No, Mask forbids 0.
            # If action invalid, env usually forces 0 (Pass).
            # But if Pass is forbidden, what do we force?
            # Random valid? Or just fallback to 0 and take penalty?
            # Let's fallback to 0 but if 0 is masked, pick random valid.
            
            valid_indices = np.where(mask)[0]
            if len(valid_indices) > 0:
                action = int(self.rng.choice(valid_indices))
            else:
                action = 0 # Should not happen if W > 0
            
        reward = 0.0
        p_exec = 0.0
        side = 0 # -1, 0, 1
        size = 0.0
        
        enable_impact = self._get_cfg("flags.enable_impact", False)
        alpha = self._get_cfg("alpha", 0.3)
        pass_penalty = self._get_cfg("pass_penalty", 0.0)
        
        if action == 0:
            # Pass
            self.consecutive_passes += 1
            # Escalating penalty still applies if they manage to pass (e.g. no money)
            current_penalty = pass_penalty * (1.0 + 0.5 * (self.consecutive_passes - 1))
            reward = -current_penalty
            p_exec = self.quote.mid
            
        else:
            self.consecutive_passes = 0
            if 1 <= action <= 10:
                # Buy
                side = 1
                size = float(action)
                p_exec = exec_price_buy(self.quote.ask, size, self.true_depths[1], alpha, enable_impact)
                reward = size * (self.true_sum - p_exec)
            
            elif 11 <= action <= 20:
                # Sell
                side = -1
                size = float(action - 10)
                p_exec = exec_price_sell(self.quote.bid, size, self.true_depths[0], alpha, enable_impact)
                reward = size * (p_exec - self.true_sum)
            
        # Update Budget
        self.W += reward
        
        # Metrics
        slippage = 0.0
        if side == 1:
            slippage = p_exec - self.quote.ask
        elif side == -1:
            slippage = self.quote.bid - p_exec
            
        self.metrics = {
            "slippage_bid": slippage if side == -1 else 0.0,
            "slippage_ask": slippage if side == 1 else 0.0,
            "fill_ratio_bid": 1.0 if side == -1 else 0.0,
            "fill_ratio_ask": 1.0 if side == 1 else 0.0
        }
        
        # Step time
        self.t += 1
        
        terminated = False
        truncated = False
        
        if self.t >= self.T:
            terminated = True
            
        stop_out_frac = self._get_cfg("stop_out", 0.2)
        if self.W < stop_out_frac * self.W0:
            terminated = True
            
        # Info for logging
        step_info = {
            "reward": reward,
            "exec_price": p_exec,
            "slippage": slippage,
            "true_sum": self.true_sum,
            "action": action,
            "budget": self.W,
            "consecutive_passes": self.consecutive_passes
        }
        
        # State info for logging (needs values from START of step usually, but t has incremented)
        # Logging usually records state at t, action at t, reward at t+1.
        # Here t is now t+1.
        state_info = {
            "t": self.t - 1,
            "W": self.W - reward, # Budget before reward
            "mu": self.mu,
            "sigma": self.sigma,
            "event": self.current_event.to_dict(),
            "true_depths": self.true_depths
        }
        
        # Create log entry
        # We pass current obs? No, we don't have obs from start of step easily available unless we stored it.
        # `make_log_entry` asks for `obs`.
        # We returned `obs` in previous step/reset.
        # For now passing None or empty.
        log_entry = make_log_entry(step_info, None, action, mask, state_info)
        
        if not terminated:
            obs_next, info_next = self._start_round()
            info_next.update(step_info)
            info_next["log"] = log_entry
            return obs_next, reward, terminated, truncated, info_next
        else:
            dummy_obs = build_obs(
                quote=self.quote,
                depths=self.disp_depths,
                metrics=self.metrics,
                hints=self.hints,
                event=self.current_event,
                state={"W": self.W, "W0": self.W0, "t": self.t, "T": self.T},
                flags={
                    "enable_events": self._get_cfg("flags.enable_events", False),
                    "enable_impact": self._get_cfg("flags.enable_impact", False)
                }
            )
            step_info["log"] = log_entry
            return dummy_obs, reward, terminated, truncated, step_info
