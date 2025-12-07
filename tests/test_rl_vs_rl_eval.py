"""
Tests for RL vs RL evaluation functionality.
"""
import pytest
import numpy as np
from mmrl.agents.ippo.ippo_agent import IPPOAgent
from mmrl.agents.mappo.mappo_agent import MAPPOAgent
from mmrl.env.spaces import get_obs_shape, ACTION_SPACE_SIZE
from mmrl.eval.compare_agents import run_rl_vs_rl_matchup, evaluate_rl_vs_rl


@pytest.fixture
def agent_config():
    """Common agent configuration."""
    return {
        "device": "cpu",
        "lr": 3e-4,
        "rollout_steps": 128,
        "train_iters": 2,
        "gamma": 0.99,
        "gae_lambda": 0.95,
    }


@pytest.fixture
def env_config():
    """Environment configuration for testing."""
    return {
        "episode_length": 5,  # Short for testing
        "W0": 500.0,
        "flags": {"enable_events": False, "enable_impact": False},
    }


def test_run_rl_vs_rl_matchup_ippo_vs_ippo(agent_config, env_config):
    """Test IPPO self-play matchup."""
    obs_dim = get_obs_shape()[0]
    act_dim = ACTION_SPACE_SIZE
    
    agent_a = IPPOAgent(obs_dim, act_dim, agent_config)
    agent_b = IPPOAgent(obs_dim, act_dim, agent_config)
    
    metrics = run_rl_vs_rl_matchup(
        agent_a, "ippo",
        agent_b, "ippo",
        env_config,
        n_episodes=2,  # Small number for testing
        seed=42
    )
    
    # Check all expected keys are present
    assert "mean_return_a" in metrics
    assert "std_return_a" in metrics
    assert "sharpe_a" in metrics
    assert "mean_return_b" in metrics
    assert "std_return_b" in metrics
    assert "sharpe_b" in metrics
    
    # Check values are reasonable
    assert isinstance(metrics["mean_return_a"], float)
    assert isinstance(metrics["mean_return_b"], float)
    assert not np.isnan(metrics["mean_return_a"])
    assert not np.isnan(metrics["mean_return_b"])


def test_run_rl_vs_rl_matchup_ippo_vs_mappo(agent_config, env_config):
    """Test IPPO vs MAPPO matchup."""
    obs_dim = get_obs_shape()[0]
    act_dim = ACTION_SPACE_SIZE
    joint_obs_dim = obs_dim * 2
    
    ippo = IPPOAgent(obs_dim, act_dim, agent_config)
    mappo = MAPPOAgent(obs_dim, joint_obs_dim, act_dim, agent_config)
    
    metrics = run_rl_vs_rl_matchup(
        ippo, "ippo",
        mappo, "mappo",
        env_config,
        n_episodes=2,
        seed=42
    )
    
    assert "mean_return_a" in metrics
    assert "mean_return_b" in metrics
    assert isinstance(metrics["sharpe_a"], float)
    assert isinstance(metrics["sharpe_b"], float)


def test_evaluate_rl_vs_rl_multiple_agents(agent_config, env_config):
    """Test evaluation of multiple RL agents."""
    obs_dim = get_obs_shape()[0]
    act_dim = ACTION_SPACE_SIZE
    joint_obs_dim = obs_dim * 2
    
    ippo = IPPOAgent(obs_dim, act_dim, agent_config)
    mappo = MAPPOAgent(obs_dim, joint_obs_dim, act_dim, agent_config)
    
    agents_dict = {
        "IPPO": (ippo, "ippo"),
        "MAPPO": (mappo, "mappo"),
    }
    
    df = evaluate_rl_vs_rl(
        agents_dict,
        env_config,
        n_episodes=2,
        seed=42,
        include_self_play=True
    )
    
    # Should have 4 rows: IPPO vs IPPO, IPPO vs MAPPO, MAPPO vs IPPO, MAPPO vs MAPPO
    assert len(df) == 4
    
    # Check columns
    assert "agent_a" in df.columns
    assert "agent_b" in df.columns
    assert "mean_return_a" in df.columns
    assert "mean_return_b" in df.columns
    
    # Check agent names
    assert "IPPO" in df["agent_a"].values
    assert "MAPPO" in df["agent_a"].values


def test_evaluate_rl_vs_rl_no_self_play(agent_config, env_config):
    """Test evaluation without self-play."""
    obs_dim = get_obs_shape()[0]
    act_dim = ACTION_SPACE_SIZE
    joint_obs_dim = obs_dim * 2
    
    ippo = IPPOAgent(obs_dim, act_dim, agent_config)
    mappo = MAPPOAgent(obs_dim, joint_obs_dim, act_dim, agent_config)
    
    agents_dict = {
        "IPPO": (ippo, "ippo"),
        "MAPPO": (mappo, "mappo"),
    }
    
    df = evaluate_rl_vs_rl(
        agents_dict,
        env_config,
        n_episodes=2,
        seed=42,
        include_self_play=False
    )
    
    # Should have 2 rows: IPPO vs MAPPO, MAPPO vs IPPO (no self-play)
    assert len(df) == 2
    
    # Verify no self-play matches
    for idx, row in df.iterrows():
        assert row["agent_a"] != row["agent_b"]


def test_evaluate_rl_vs_rl_single_agent(agent_config, env_config):
    """Test evaluation with single agent (self-play only)."""
    obs_dim = get_obs_shape()[0]
    act_dim = ACTION_SPACE_SIZE
    
    ippo = IPPOAgent(obs_dim, act_dim, agent_config)
    
    agents_dict = {
        "IPPO": (ippo, "ippo"),
    }
    
    df = evaluate_rl_vs_rl(
        agents_dict,
        env_config,
        n_episodes=2,
        seed=42,
        include_self_play=True
    )
    
    # Should have 1 row: IPPO vs IPPO
    assert len(df) == 1
    assert df.iloc[0]["agent_a"] == "IPPO"
    assert df.iloc[0]["agent_b"] == "IPPO"

