"""
Tests for EV Realistic baseline.

Tests that the EV Realistic agent:
1. Correctly decodes events and hints from observations
2. Computes fair value using only observable information
3. Properly handles market impact calculations
4. Does NOT use privileged information (mu from info, true_sum, true_depths)
"""
import pytest
import numpy as np
from unittest.mock import patch

from mmrl.env.single_env import SingleCardEnv
from mmrl.env.spaces import build_obs, get_obs_shape
from mmrl.env.events import Event, EVENT_NONE, EVENT_GE10, EVENT_EVEN
from mmrl.env.quotes import Quote
from mmrl.baselines.ev_realistic import (
    EVRealisticAgent,
    act_ev_realistic,
    decode_event_from_obs,
    decode_hints_from_obs,
    compute_fair_value_from_obs,
    compute_expected_overflow,
    DEFAULT_LIQUIDITY_CFG
)


@pytest.fixture
def basic_env():
    """Create a basic environment for testing."""
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
def sample_obs():
    """Create a sample observation for testing."""
    quote = Quote(mid=25.0, spread=2.0, bid=24.0, ask=26.0)
    depths = (8.0, 8.0)  # displayed depths
    metrics = {"slippage_bid": 0.0, "slippage_ask": 0.0, 
               "fill_ratio_bid": 0.0, "fill_ratio_ask": 0.0}
    hints = [10, 8]  # Two revealed hints
    event = Event(type=EVENT_NONE)
    state = {"W": 500.0, "W0": 500.0, "t": 0, "T": 10}
    flags = {"enable_events": False, "enable_impact": False}
    
    return build_obs(quote, depths, metrics, hints, event, state, flags)


class TestDecodeEvent:
    """Tests for event decoding from observation."""
    
    def test_decode_none_event(self, sample_obs):
        """Test decoding EVENT_NONE."""
        event = decode_event_from_obs(sample_obs)
        assert event.type == EVENT_NONE
    
    def test_decode_ge10_event(self):
        """Test decoding ge10_only event."""
        quote = Quote(mid=35.0, spread=2.0, bid=34.0, ask=36.0)
        event = Event(type=EVENT_GE10)
        obs = build_obs(
            quote, (8.0, 8.0), {}, [], event,
            {"W": 500.0, "W0": 500.0, "t": 0, "T": 10},
            {"enable_events": True, "enable_impact": False}
        )
        decoded = decode_event_from_obs(obs)
        assert decoded.type == EVENT_GE10
    
    def test_decode_even_event(self):
        """Test decoding even_only event."""
        quote = Quote(mid=24.0, spread=2.0, bid=23.0, ask=25.0)
        event = Event(type=EVENT_EVEN)
        obs = build_obs(
            quote, (8.0, 8.0), {}, [], event,
            {"W": 500.0, "W0": 500.0, "t": 0, "T": 10},
            {"enable_events": True, "enable_impact": False}
        )
        decoded = decode_event_from_obs(obs)
        assert decoded.type == EVENT_EVEN


class TestDecodeHints:
    """Tests for hint decoding from observation."""
    
    def test_decode_no_hints(self):
        """Test decoding when no hints are revealed."""
        quote = Quote(mid=25.0, spread=2.0, bid=24.0, ask=26.0)
        obs = build_obs(
            quote, (8.0, 8.0), {}, [], Event(type=EVENT_NONE),
            {"W": 500.0, "W0": 500.0, "t": 0, "T": 10},
            {"enable_events": False, "enable_impact": False}
        )
        hints = decode_hints_from_obs(obs)
        assert hints == []
    
    def test_decode_single_hint(self):
        """Test decoding a single hint."""
        quote = Quote(mid=25.0, spread=2.0, bid=24.0, ask=26.0)
        obs = build_obs(
            quote, (8.0, 8.0), {}, [7], Event(type=EVENT_NONE),
            {"W": 500.0, "W0": 500.0, "t": 0, "T": 10},
            {"enable_events": False, "enable_impact": False}
        )
        hints = decode_hints_from_obs(obs)
        assert hints == [7]
    
    def test_decode_multiple_hints(self):
        """Test decoding multiple hints."""
        quote = Quote(mid=25.0, spread=2.0, bid=24.0, ask=26.0)
        obs = build_obs(
            quote, (8.0, 8.0), {}, [5, 10, 14], Event(type=EVENT_NONE),
            {"W": 500.0, "W0": 500.0, "t": 0, "T": 10},
            {"enable_events": False, "enable_impact": False}
        )
        hints = decode_hints_from_obs(obs)
        assert sorted(hints) == [5, 10, 14]
    
    def test_decode_duplicate_hints(self):
        """Test decoding when same rank appears multiple times."""
        quote = Quote(mid=25.0, spread=2.0, bid=24.0, ask=26.0)
        obs = build_obs(
            quote, (8.0, 8.0), {}, [8, 8], Event(type=EVENT_NONE),
            {"W": 500.0, "W0": 500.0, "t": 0, "T": 10},
            {"enable_events": False, "enable_impact": False}
        )
        hints = decode_hints_from_obs(obs)
        assert hints == [8, 8]


class TestFairValueComputation:
    """Tests for fair value computation from observations."""
    
    def test_fair_value_no_hints(self):
        """Test fair value with no hints revealed."""
        quote = Quote(mid=25.0, spread=2.0, bid=24.0, ask=26.0)
        obs = build_obs(
            quote, (8.0, 8.0), {}, [], Event(type=EVENT_NONE),
            {"W": 500.0, "W0": 500.0, "t": 0, "T": 10},
            {"enable_events": False, "enable_impact": False}
        )
        mu, sigma = compute_fair_value_from_obs(obs)
        
        # With no hints and full deck, expected sum of 3 cards ≈ 3 * 8 = 24
        # (mean of 2..14 is 8)
        assert 20 < mu < 28  # Reasonable range
        assert sigma > 0  # Should have uncertainty
    
    def test_fair_value_with_hints(self):
        """Test fair value when hints are revealed."""
        quote = Quote(mid=30.0, spread=2.0, bid=29.0, ask=31.0)
        # Reveal two high cards
        obs = build_obs(
            quote, (8.0, 8.0), {}, [14, 13], Event(type=EVENT_NONE),  # A + K revealed
            {"W": 500.0, "W0": 500.0, "t": 0, "T": 10},
            {"enable_events": False, "enable_impact": False}
        )
        mu, sigma = compute_fair_value_from_obs(obs)
        
        # With A(14) and K(13) revealed, mu = 27 + E[remaining]
        # Remaining is one card from deck minus {14, 13}
        # Mean of remaining deck ≈ 7.8
        assert 33 < mu < 37  # 27 + 7 to 27 + 9
        assert sigma > 0
    
    def test_fair_value_all_hints(self):
        """Test fair value when all cards are revealed."""
        quote = Quote(mid=30.0, spread=2.0, bid=29.0, ask=31.0)
        # All 3 cards revealed
        obs = build_obs(
            quote, (8.0, 8.0), {}, [5, 7, 10], Event(type=EVENT_NONE),
            {"W": 500.0, "W0": 500.0, "t": 0, "T": 10},
            {"enable_events": False, "enable_impact": False}
        )
        mu, sigma = compute_fair_value_from_obs(obs)
        
        # Exact sum is known: 5 + 7 + 10 = 22
        assert abs(mu - 22) < 0.01
        assert sigma < 0.01  # No uncertainty
    
    def test_fair_value_with_ge10_event(self):
        """Test fair value with ge10_only event."""
        quote = Quote(mid=35.0, spread=2.0, bid=34.0, ask=36.0)
        obs = build_obs(
            quote, (8.0, 8.0), {}, [], Event(type=EVENT_GE10),
            {"W": 500.0, "W0": 500.0, "t": 0, "T": 10},
            {"enable_events": True, "enable_impact": False}
        )
        mu, sigma = compute_fair_value_from_obs(obs)
        
        # With ge10_only, cards are {10, 11, 12, 13, 14}
        # Mean = 12, expected sum of 3 = 36
        assert 34 < mu < 38
        assert sigma > 0


class TestExpectedOverflow:
    """Tests for expected overflow calculation (for impact)."""
    
    def test_no_overflow_known_depth(self):
        """Test when order size < known depth."""
        # D_disp < L_cap means L_true = D_disp exactly
        overflow = compute_expected_overflow(
            order_size=5.0,
            D_disp=8.0,  # Less than cap
            L_cap=10.0,
            liq_cfg=DEFAULT_LIQUIDITY_CFG
        )
        # Order 5, depth 8 → no overflow
        assert overflow == 0.0
    
    def test_overflow_known_depth(self):
        """Test when order size > known depth."""
        overflow = compute_expected_overflow(
            order_size=10.0,
            D_disp=6.0,  # Less than cap, so L_true = 6.0
            L_cap=10.0,
            liq_cfg=DEFAULT_LIQUIDITY_CFG
        )
        # Order 10, depth 6 → overflow = 4
        assert abs(overflow - 4.0) < 0.01
    
    def test_no_overflow_at_cap(self):
        """Test no overflow when order size <= cap and D_disp = cap."""
        overflow = compute_expected_overflow(
            order_size=5.0,
            D_disp=10.0,  # At cap
            L_cap=10.0,
            liq_cfg=DEFAULT_LIQUIDITY_CFG
        )
        # Order 5, L_true >= 10, so (5-L)+ = 0 always
        assert overflow == 0.0
    
    def test_positive_overflow_at_cap(self):
        """Test positive expected overflow when D_disp = cap and order > cap."""
        overflow = compute_expected_overflow(
            order_size=15.0,
            D_disp=10.0,  # At cap
            L_cap=10.0,
            liq_cfg=DEFAULT_LIQUIDITY_CFG
        )
        # Order 15, L_true >= 10 (truncated), expect some overflow
        assert overflow > 0.0
        # But less than worst case (15 - 10 = 5) since E[L] > 10
        assert overflow < 5.0


class TestEVRealisticAgent:
    """Tests for the EVRealisticAgent class."""
    
    def test_agent_creation(self):
        """Test agent can be created."""
        agent = EVRealisticAgent()
        assert agent is not None
        assert agent.alpha == 0.15
    
    def test_agent_custom_alpha(self):
        """Test agent with custom alpha."""
        agent = EVRealisticAgent(alpha=0.3)
        assert agent.alpha == 0.3
    
    def test_action_valid(self, basic_env):
        """Test that agent returns valid actions."""
        agent = EVRealisticAgent()
        obs, info = basic_env.reset(seed=42)
        
        for _ in range(10):
            mask = info["mask"]
            action = agent.act(obs, mask, info=info)
            
            # Action should be valid
            assert 0 <= action <= 20
            assert mask[action], f"Action {action} not in valid mask"
            
            obs, _, term, trunc, info = basic_env.step(action)
            if term or trunc:
                obs, info = basic_env.reset()
    
    def test_does_not_use_privileged_info(self, basic_env):
        """Test that agent does NOT use privileged info from info dict."""
        agent = EVRealisticAgent()
        obs, info = basic_env.reset(seed=42)
        mask = info["mask"]
        
        # Get action with correct info
        action1 = agent.act(obs, mask, info=info)
        
        # Get action with wrong/corrupted info
        fake_info = {
            "mu": 999.0,  # Completely wrong
            "true_sum": 999,  # Wrong
            "true_depths": (999.0, 999.0),  # Wrong
            "mask": mask
        }
        action2 = agent.act(obs, mask, info=fake_info)
        
        # Actions should be the same because EV Realistic ignores info
        assert action1 == action2
    
    def test_get_fair_value(self, sample_obs):
        """Test get_fair_value method."""
        agent = EVRealisticAgent()
        mu, sigma = agent.get_fair_value(sample_obs)
        
        assert isinstance(mu, float)
        assert isinstance(sigma, float)
        assert not np.isnan(mu)
        assert not np.isnan(sigma)
        assert sigma >= 0
    
    def test_buy_when_undervalued(self):
        """Test that agent buys when fair value > ask."""
        agent = EVRealisticAgent()
        
        # Create obs where all hints reveal high-value cards
        # This should make mu > ask, triggering a buy
        quote = Quote(mid=30.0, spread=2.0, bid=29.0, ask=31.0)
        # Reveal A + K = 27, remaining card mean ≈ 8 → total ≈ 35 > 31
        obs = build_obs(
            quote, (8.0, 8.0), {}, [14, 13], Event(type=EVENT_NONE),
            {"W": 500.0, "W0": 500.0, "t": 0, "T": 10},
            {"enable_events": False, "enable_impact": False}
        )
        mask = np.ones(21, dtype=bool)
        
        action = agent.act(obs, mask)
        
        # Should be a buy action (1-10)
        assert 1 <= action <= 10, f"Expected buy action, got {action}"
    
    def test_sell_when_overvalued(self):
        """Test that agent sells when fair value < bid."""
        agent = EVRealisticAgent()
        
        # Create obs where hints reveal low-value cards
        # This should make mu < bid, triggering a sell
        quote = Quote(mid=20.0, spread=2.0, bid=19.0, ask=21.0)
        # Reveal 2 + 3 = 5, remaining mean ≈ 8 → total ≈ 13 < 19
        obs = build_obs(
            quote, (8.0, 8.0), {}, [2, 3], Event(type=EVENT_NONE),
            {"W": 500.0, "W0": 500.0, "t": 0, "T": 10},
            {"enable_events": False, "enable_impact": False}
        )
        mask = np.ones(21, dtype=bool)
        
        action = agent.act(obs, mask)
        
        # Should be a sell action (11-20)
        assert 11 <= action <= 20, f"Expected sell action, got {action}"
    
    def test_pass_when_no_edge(self):
        """Test that agent passes when fair value ≈ mid."""
        agent = EVRealisticAgent()
        
        # Set quote mid close to expected fair value (no hints, full deck)
        # Expected fair value ≈ 24 (mean of full deck)
        quote = Quote(mid=24.0, spread=2.0, bid=23.0, ask=25.0)
        obs = build_obs(
            quote, (8.0, 8.0), {}, [], Event(type=EVENT_NONE),
            {"W": 500.0, "W0": 500.0, "t": 0, "T": 10},
            {"enable_events": False, "enable_impact": False}
        )
        mask = np.ones(21, dtype=bool)
        
        action = agent.act(obs, mask)
        
        # With fair value ≈ 24, bid=23, ask=25, there's no edge
        # Should pass
        assert action == 0, f"Expected pass (0), got {action}"


class TestIntegrationWithEnv:
    """Integration tests with actual environment."""
    
    def test_full_episode_no_crash(self, basic_env):
        """Test that agent can complete a full episode without crashing."""
        agent = EVRealisticAgent()
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
        # Agent should complete without errors
    
    def test_full_episode_with_impact(self, impact_env):
        """Test agent with impact enabled."""
        agent = EVRealisticAgent(alpha=0.15)
        obs, info = impact_env.reset(seed=42)
        
        done = False
        while not done:
            mask = info["mask"]
            action = agent.act(obs, mask, info=info)
            obs, reward, term, trunc, info = impact_env.step(action)
            done = term or trunc
        
        # Should complete without errors
    
    def test_multiple_episodes(self, basic_env):
        """Test agent over multiple episodes for consistency."""
        agent = EVRealisticAgent()
        
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
        
        # Check we got reasonable returns (not all zero, some variance)
        returns = np.array(returns)
        assert len(returns) == 10
        # Returns should have some variance (agent is making decisions)
        assert np.std(returns) > 0 or np.mean(returns) != 0


class TestComparisonWithOracle:
    """Tests comparing EV Realistic with EV Oracle."""
    
    def test_different_from_oracle_without_hints(self, basic_env):
        """
        Test that EV Realistic may differ from Oracle when no hints.
        
        EV Oracle knows the exact mu from environment, while
        EV Realistic computes its own estimate which may differ
        due to quote noise.
        """
        from mmrl.baselines.ev_oracle import EVOracleAgent
        
        realistic = EVRealisticAgent()
        oracle = EVOracleAgent()
        
        obs, info = basic_env.reset(seed=42)
        mask = info["mask"]
        
        # Get fair value computed by realistic
        mu_realistic, _ = realistic.get_fair_value(obs)
        
        # Oracle uses mu from info
        mu_oracle = info["mu"]
        
        # They should be close but not necessarily identical
        # (realistic uses its own Bayesian computation)
        # The difference comes from the fact that both compute
        # the same posterior, but the quote was generated with noise
        # So the difference should be small
        assert abs(mu_realistic - mu_oracle) < 5.0  # Reasonable tolerance

