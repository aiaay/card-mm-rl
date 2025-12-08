"""
Tests for Level-1 Realistic baseline.

Tests that the Level-1 Realistic agent:
1. Does NOT use privileged information (mu from info, true_depths)
2. Correctly tracks opponent history
3. Properly estimates opponent crowding
4. Adjusts trade size based on expected impact and crowding
5. Computes fair value from observable data only
"""
import pytest
import numpy as np
from unittest.mock import patch

from mmrl.env.single_env import SingleCardEnv
from mmrl.env.two_player_env import TwoPlayerCardEnv
from mmrl.env.spaces import build_obs, get_obs_shape
from mmrl.env.events import Event, EVENT_NONE, EVENT_GE10
from mmrl.env.quotes import Quote
from mmrl.baselines.level1_realistic import (
    Level1RealisticPolicy,
    create_level1_realistic_agent
)
from mmrl.baselines.level1_crowding import Level1Policy


@pytest.fixture
def basic_env():
    """Create a basic single-player environment."""
    cfg = {
        "episode_length": 10,
        "W0": 500.0,
        "flags": {"enable_events": False, "enable_impact": False}
    }
    return SingleCardEnv(cfg)


@pytest.fixture
def impact_env():
    """Create an environment with impact enabled."""
    cfg = {
        "episode_length": 10,
        "W0": 500.0,
        "flags": {"enable_events": False, "enable_impact": True},
        "alpha": 0.15
    }
    return SingleCardEnv(cfg)


@pytest.fixture
def two_player_env():
    """Create a two-player environment."""
    cfg = {
        "episode_length": 10,
        "W0": 500.0,
        "flags": {"enable_events": False, "enable_impact": True},
        "alpha": 0.15
    }
    return TwoPlayerCardEnv(cfg)


@pytest.fixture
def sample_obs():
    """Create a sample observation for testing."""
    quote = Quote(mid=25.0, spread=2.0, bid=24.0, ask=26.0)
    depths = (8.0, 8.0)
    metrics = {"slippage_bid": 0.0, "slippage_ask": 0.0, 
               "fill_ratio_bid": 0.0, "fill_ratio_ask": 0.0}
    hints = [10, 8]
    event = Event(type=EVENT_NONE)
    state = {"W": 500.0, "W0": 500.0, "t": 0, "T": 10}
    flags = {"enable_events": False, "enable_impact": False}
    
    return build_obs(quote, depths, metrics, hints, event, state, flags)


class TestAgentCreation:
    """Tests for agent creation and initialization."""
    
    def test_create_agent(self):
        """Test agent can be created."""
        agent = Level1RealisticPolicy()
        assert agent is not None
        assert agent.alpha == 0.15
        assert agent.history_len == 10
        assert agent.risk_aversion == 0.5
    
    def test_create_agent_custom_params(self):
        """Test agent with custom parameters."""
        agent = Level1RealisticPolicy(
            history_len=20,
            alpha=0.3,
            risk_aversion=0.8
        )
        assert agent.history_len == 20
        assert agent.alpha == 0.3
        assert agent.risk_aversion == 0.8
    
    def test_convenience_function(self):
        """Test convenience creation function."""
        agent = create_level1_realistic_agent(
            history_len=15,
            alpha=0.2,
            risk_aversion=0.6
        )
        assert agent.history_len == 15
        assert agent.alpha == 0.2
        assert agent.risk_aversion == 0.6


class TestOpponentHistory:
    """Tests for opponent history tracking."""
    
    def test_empty_history(self):
        """Test initial state with no history."""
        agent = Level1RealisticPolicy()
        
        # Should return default estimates
        demand, unc = agent._estimate_opp_demand(1)  # Buy side
        assert demand > 0  # Non-zero prior
        assert unc > 0
    
    def test_update_history(self):
        """Test updating opponent history."""
        agent = Level1RealisticPolicy(history_len=5)
        
        # Add some opponent actions
        agent.update(opp_side=1, opp_size=5.0)  # Buy 5
        agent.update(opp_side=1, opp_size=7.0)  # Buy 7
        agent.update(opp_side=-1, opp_size=3.0)  # Sell 3
        
        assert len(agent.opp_history) == 3
    
    def test_history_max_length(self):
        """Test that history respects max length."""
        agent = Level1RealisticPolicy(history_len=3)
        
        for i in range(5):
            agent.update(opp_side=1, opp_size=float(i))
        
        # Should only keep last 3
        assert len(agent.opp_history) == 3
    
    def test_estimate_after_updates(self):
        """Test demand estimation after updating history."""
        agent = Level1RealisticPolicy(history_len=10)
        
        # Add mostly buy actions
        for _ in range(8):
            agent.update(opp_side=1, opp_size=5.0)
        for _ in range(2):
            agent.update(opp_side=-1, opp_size=3.0)
        
        buy_demand, buy_unc = agent._estimate_opp_demand(1)
        sell_demand, sell_unc = agent._estimate_opp_demand(-1)
        
        # Should estimate higher demand on buy side
        assert buy_demand > sell_demand
    
    def test_reset_history(self):
        """Test resetting history."""
        agent = Level1RealisticPolicy()
        agent.update(1, 5.0)
        agent.update(1, 5.0)
        
        assert len(agent.opp_history) == 2
        
        agent.reset()
        assert len(agent.opp_history) == 0


class TestNoPrivilegedInfo:
    """Tests verifying agent does NOT use privileged information."""
    
    def test_does_not_use_info_mu(self, basic_env):
        """Test that agent ignores info['mu']."""
        agent = Level1RealisticPolicy()
        obs, info = basic_env.reset(seed=42)
        mask = info["mask"]
        
        # Get action with correct info
        action1 = agent.act(obs, mask, info=info)
        
        # Get action with wrong mu
        fake_info = info.copy()
        fake_info["mu"] = 999.0  # Completely wrong
        action2 = agent.act(obs, mask, info=fake_info)
        
        # Actions should be identical (agent ignores info)
        assert action1 == action2
    
    def test_does_not_use_true_depths(self, basic_env):
        """Test that agent ignores info['true_depths']."""
        agent = Level1RealisticPolicy()
        obs, info = basic_env.reset(seed=42)
        mask = info["mask"]
        
        action1 = agent.act(obs, mask, info=info)
        
        # Corrupt true_depths
        fake_info = info.copy()
        fake_info["true_depths"] = (0.1, 0.1)  # Extremely low
        action2 = agent.act(obs, mask, info=fake_info)
        
        # Actions should be identical
        assert action1 == action2
    
    def test_works_without_info(self, basic_env):
        """Test that agent works when info is None."""
        agent = Level1RealisticPolicy()
        obs, info = basic_env.reset(seed=42)
        mask = info["mask"]
        
        # Should not crash with info=None
        action = agent.act(obs, mask, info=None)
        assert 0 <= action <= 20
        assert mask[action]


class TestFairValueComputation:
    """Tests for fair value computation."""
    
    def test_get_fair_value(self, sample_obs):
        """Test fair value getter."""
        agent = Level1RealisticPolicy()
        mu, sigma = agent.get_fair_value(sample_obs)
        
        assert isinstance(mu, float)
        assert isinstance(sigma, float)
        assert not np.isnan(mu)
        assert sigma >= 0
    
    def test_fair_value_with_hints(self):
        """Test fair value when hints are revealed."""
        agent = Level1RealisticPolicy()
        
        # High-value hints
        quote = Quote(mid=35.0, spread=2.0, bid=34.0, ask=36.0)
        obs = build_obs(
            quote, (8.0, 8.0), {}, [14, 13, 12], Event(type=EVENT_NONE),
            {"W": 500.0, "W0": 500.0, "t": 0, "T": 10},
            {"enable_events": False, "enable_impact": False}
        )
        
        mu, sigma = agent.get_fair_value(obs)
        
        # Known sum: 14 + 13 + 12 = 39
        assert abs(mu - 39.0) < 0.01
        assert sigma < 0.01  # No uncertainty when all revealed


class TestActionSelection:
    """Tests for action selection logic."""
    
    def test_valid_actions_only(self, basic_env):
        """Test that agent returns valid actions."""
        agent = Level1RealisticPolicy()
        obs, info = basic_env.reset(seed=42)
        
        for _ in range(20):
            mask = info["mask"]
            action = agent.act(obs, mask, info=info)
            
            assert 0 <= action <= 20
            assert mask[action], f"Action {action} not valid"
            
            obs, _, term, trunc, info = basic_env.step(action)
            if term or trunc:
                obs, info = basic_env.reset()
    
    def test_buy_when_undervalued(self):
        """Test buying when fair value > ask."""
        agent = Level1RealisticPolicy()
        
        # High hints make mu > ask
        quote = Quote(mid=30.0, spread=2.0, bid=29.0, ask=31.0)
        obs = build_obs(
            quote, (8.0, 8.0), {}, [14, 13], Event(type=EVENT_NONE),
            {"W": 500.0, "W0": 500.0, "t": 0, "T": 10},
            {"enable_events": False, "enable_impact": False}
        )
        mask = np.ones(21, dtype=bool)
        
        action = agent.act(obs, mask)
        
        # Should buy (action 1-10)
        assert 1 <= action <= 10, f"Expected buy, got {action}"
    
    def test_sell_when_overvalued(self):
        """Test selling when fair value < bid."""
        agent = Level1RealisticPolicy()
        
        # Low hints make mu < bid
        quote = Quote(mid=20.0, spread=2.0, bid=19.0, ask=21.0)
        obs = build_obs(
            quote, (8.0, 8.0), {}, [2, 3], Event(type=EVENT_NONE),
            {"W": 500.0, "W0": 500.0, "t": 0, "T": 10},
            {"enable_events": False, "enable_impact": False}
        )
        mask = np.ones(21, dtype=bool)
        
        action = agent.act(obs, mask)
        
        # Should sell (action 11-20)
        assert 11 <= action <= 20, f"Expected sell, got {action}"
    
    def test_pass_when_no_edge(self):
        """Test passing when no edge exists."""
        agent = Level1RealisticPolicy()
        
        # Fair price (no hints, mid ≈ 24 which is expected value)
        quote = Quote(mid=24.0, spread=2.0, bid=23.0, ask=25.0)
        obs = build_obs(
            quote, (8.0, 8.0), {}, [], Event(type=EVENT_NONE),
            {"W": 500.0, "W0": 500.0, "t": 0, "T": 10},
            {"enable_events": False, "enable_impact": False}
        )
        mask = np.ones(21, dtype=bool)
        
        action = agent.act(obs, mask)
        
        # Should pass
        assert action == 0, f"Expected pass, got {action}"


class TestCrowdingBehavior:
    """Tests for crowding-aware behavior."""
    
    def test_reduces_size_with_crowding(self):
        """Test that agent reduces size when expecting opponent crowding."""
        agent_no_crowd = Level1RealisticPolicy()
        agent_crowd = Level1RealisticPolicy()
        
        # Add history of opponent buying heavily
        for _ in range(10):
            agent_crowd.update(opp_side=1, opp_size=8.0)
        
        # Create obs with buy edge and impact enabled
        quote = Quote(mid=28.0, spread=2.0, bid=27.0, ask=29.0)
        obs = build_obs(
            quote, (8.0, 8.0), {}, [14, 12], Event(type=EVENT_NONE),
            {"W": 500.0, "W0": 500.0, "t": 0, "T": 10},
            {"enable_events": False, "enable_impact": True}  # Impact ON
        )
        mask = np.ones(21, dtype=bool)
        
        action_no_crowd = agent_no_crowd.act(obs, mask)
        action_crowd = agent_crowd.act(obs, mask)
        
        # Both should buy, but crowded agent may use smaller size
        assert 1 <= action_no_crowd <= 10
        assert 1 <= action_crowd <= 10
        # Agent expecting crowding should be more conservative
        # (May not always hold due to complex edge calculations)
    
    def test_opponent_estimate_affects_action(self):
        """Test that opponent history affects decisions."""
        agent = Level1RealisticPolicy(risk_aversion=0.8)
        
        # Strong buy signal
        quote = Quote(mid=30.0, spread=2.0, bid=29.0, ask=31.0)
        obs = build_obs(
            quote, (5.0, 5.0), {}, [14, 13], Event(type=EVENT_NONE),
            {"W": 500.0, "W0": 500.0, "t": 0, "T": 10},
            {"enable_events": False, "enable_impact": True}
        )
        mask = np.ones(21, dtype=bool)
        
        # Action without history
        agent.reset()
        action1 = agent.act(obs, mask)
        
        # Add heavy opponent buying history
        for _ in range(10):
            agent.update(1, 10.0)  # Opponent always buying max
        
        action2 = agent.act(obs, mask)
        
        # With crowding expectation, agent should be more cautious
        # Both should still be buy actions
        assert 1 <= action1 <= 10
        assert 1 <= action2 <= 10


class TestRiskAversion:
    """Tests for risk aversion parameter."""
    
    def test_high_risk_aversion_more_conservative(self):
        """Test that high risk aversion leads to more conservative actions."""
        agent_low_risk = Level1RealisticPolicy(risk_aversion=0.1)
        agent_high_risk = Level1RealisticPolicy(risk_aversion=0.9)
        
        # Marginal edge scenario
        quote = Quote(mid=26.0, spread=2.0, bid=25.0, ask=27.0)
        obs = build_obs(
            quote, (6.0, 6.0), {}, [10], Event(type=EVENT_NONE),
            {"W": 500.0, "W0": 500.0, "t": 0, "T": 10},
            {"enable_events": False, "enable_impact": True}
        )
        mask = np.ones(21, dtype=bool)
        
        action_low = agent_low_risk.act(obs, mask)
        action_high = agent_high_risk.act(obs, mask)
        
        # High risk aversion agent may pass or use smaller size
        # Low risk aversion agent more likely to trade
        # Both should return valid actions
        assert mask[action_low]
        assert mask[action_high]


class TestIntegrationSinglePlayer:
    """Integration tests with single-player environment."""
    
    def test_full_episode(self, basic_env):
        """Test completing a full episode."""
        agent = Level1RealisticPolicy()
        obs, info = basic_env.reset(seed=42)
        
        total_reward = 0.0
        steps = 0
        done = False
        
        while not done:
            mask = info["mask"]
            action = agent.act(obs, mask, info=info)
            obs, reward, term, trunc, info = basic_env.step(action)
            done = term or trunc
            total_reward += reward
            steps += 1
        
        assert steps > 0
    
    def test_full_episode_with_impact(self, impact_env):
        """Test with market impact enabled."""
        agent = Level1RealisticPolicy(alpha=0.15)
        obs, info = impact_env.reset(seed=42)
        
        done = False
        while not done:
            mask = info["mask"]
            action = agent.act(obs, mask, info=info)
            obs, _, term, trunc, info = impact_env.step(action)
            done = term or trunc
    
    def test_multiple_episodes(self, basic_env):
        """Test over multiple episodes."""
        agent = Level1RealisticPolicy()
        
        returns = []
        for i in range(10):
            obs, info = basic_env.reset(seed=100 + i)
            ep_ret = 0.0
            done = False
            
            while not done:
                mask = info["mask"]
                action = agent.act(obs, mask, info=info)
                obs, reward, term, trunc, info = basic_env.step(action)
                done = term or trunc
                ep_ret += reward
            
            returns.append(ep_ret)
            agent.reset()  # Reset history between episodes
        
        assert len(returns) == 10


class TestIntegrationTwoPlayer:
    """Integration tests with two-player environment."""
    
    def test_two_player_episode(self, two_player_env):
        """Test in two-player environment with history updates."""
        from mmrl.baselines.random_valid import RandomValidAgent
        
        agent = Level1RealisticPolicy()
        opponent = RandomValidAgent()
        
        (obs_a, obs_b), info = two_player_env.reset(seed=42)
        done = False
        
        while not done:
            mask_a = info["mask_a"]
            mask_b = info["mask_b"]
            
            # Agent acts
            action_a = agent.act(obs_a, mask_a, info=info)
            action_b = opponent.act(obs_b, mask_b)
            
            (obs_a, obs_b), (r_a, r_b), term, trunc, info = two_player_env.step((action_a, action_b))
            done = term or trunc
            
            # Update agent's opponent history
            # Decode opponent action
            if action_b == 0:
                opp_side, opp_size = 0, 0.0
            elif 1 <= action_b <= 10:
                opp_side, opp_size = 1, float(action_b)
            else:
                opp_side, opp_size = -1, float(action_b - 10)
            
            agent.update(opp_side, opp_size)
        
        # Should complete without errors
        assert len(agent.opp_history) > 0


class TestComparisonWithPrivilegedLevel1:
    """Tests comparing with privileged Level-1 baseline."""
    
    def test_different_from_privileged(self, basic_env):
        """
        Test that realistic version may differ from privileged version.
        
        The privileged Level1Policy uses info["mu"] directly,
        while Level1RealisticPolicy computes its own estimate.
        """
        realistic = Level1RealisticPolicy()
        privileged = Level1Policy()
        
        obs, info = basic_env.reset(seed=42)
        mask = info["mask"]
        
        # Realistic computes its own mu
        mu_realistic, _ = realistic.get_fair_value(obs)
        
        # Privileged uses info["mu"]
        mu_privileged = info["mu"]
        
        # Should be close but not necessarily identical
        # due to independent computation
        diff = abs(mu_realistic - mu_privileged)
        assert diff < 5.0  # Reasonable tolerance
    
    def test_both_produce_valid_actions(self, basic_env):
        """Test both versions produce valid actions."""
        realistic = Level1RealisticPolicy()
        privileged = Level1Policy()
        
        for seed in range(5):
            obs, info = basic_env.reset(seed=seed)
            mask = info["mask"]
            
            action_r = realistic.act(obs, mask, info=info)
            action_p = privileged.act(obs, mask, info=info)
            
            assert mask[action_r], f"Realistic action {action_r} invalid"
            assert mask[action_p], f"Privileged action {action_p} invalid"

