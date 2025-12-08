import argparse
import os
from typing import Dict, Any, List

import numpy as np
import torch
import pandas as pd

from mmrl.env.single_env import SingleCardEnv
from mmrl.env.two_player_env import TwoPlayerCardEnv
from mmrl.env.spaces import get_obs_shape, ACTION_SPACE_SIZE

from mmrl.agents.dqn.dqn_agent import DQNAgent
from mmrl.agents.ippo.ippo_agent import IPPOAgent
from mmrl.agents.mappo.mappo_agent import MAPPOAgent

from mmrl.baselines.random_valid import RandomValidAgent
from mmrl.baselines.ev_oracle import EVOracleAgent
from mmrl.baselines.ev_realistic import EVRealisticAgent
from mmrl.baselines.level1_crowding import Level1Policy
from mmrl.baselines.level1_realistic import Level1RealisticPolicy


# -------- Helpers --------

def decode_action(a: int):
    if a == 0:
        return 0, 0.0
    if 1 <= a <= 10:
        return 1, float(a)
    if 11 <= a <= 20:
        return -1, float(a - 10)
    return 0, 0.0


def load_dqn(model_path: str, device: str):
    obs_dim = get_obs_shape()[0]
    act_dim = ACTION_SPACE_SIZE
    agent_cfg = {
        "device": device,
        "gamma": 0.99,
        "batch_size": 32,
        "lr": 1e-3,
        "epsilon_start": 0.0,  # no exploration at eval
        "epsilon_min": 0.0,
        "epsilon_decay": 1.0,
        "buffer_size": 1,  # unused for eval
        "hidden_dims": [64, 64],
    }
    agent = DQNAgent(obs_dim, act_dim, agent_cfg)
    state = torch.load(model_path, map_location=device)
    agent.q_net.load_state_dict(state)
    agent.epsilon = 0.0
    return agent


def load_ippo(model_path: str, device: str):
    obs_dim = get_obs_shape()[0]
    act_dim = ACTION_SPACE_SIZE
    agent_cfg = {
        "device": device,
        "lr": 3e-4,
        "gamma": 0.99,
        "rollout_steps": 2048,
        "train_iters": 4,
    }
    agent = IPPOAgent(obs_dim, act_dim, agent_cfg)
    state = torch.load(model_path, map_location=device)
    agent.ac.load_state_dict(state)
    return agent


def load_mappo(model_path: str, device: str):
    obs_dim = get_obs_shape()[0]
    joint_obs_dim = obs_dim * 2
    act_dim = ACTION_SPACE_SIZE
    agent_cfg = {
        "device": device,
        "lr": 3e-4,
        "gamma": 0.99,
        "rollout_steps": 2048,
        "train_iters": 4,
    }
    agent = MAPPOAgent(obs_dim, joint_obs_dim, act_dim, agent_cfg)
    state = torch.load(model_path, map_location=device)
    agent.ac.load_state_dict(state)
    return agent


# -------- Single-player evaluation (DQN + baselines) --------

def act_with_info(agent, obs, mask, info):
    # DQNAgent: act(obs, mask, eval_mode=False)
    if hasattr(agent, "act"):
        try:
            return agent.act(obs, mask, eval_mode=True)
        except TypeError:
            # For baselines expecting info (EVOracle, Level1)
            try:
                return agent.act(obs, mask, info=info)
            except TypeError:
                return agent.act(obs, mask)
    # Fallback if callable
    return agent(obs, mask)


def evaluate_single_agents(agents: Dict[str, Any], cfg: Dict[str, Any], n_episodes: int, seed: int):
    env = SingleCardEnv(cfg)
    rng = np.random.RandomState(seed)
    results = []

    for name, agent in agents.items():
        returns = []
        ruin = 0
        for ep in range(n_episodes):
            obs, info = env.reset(seed=int(rng.randint(0, 1e9)))
            done = False
            ep_ret = 0.0
            W = env.W0
            stop_out_frac = env._get_cfg("stop_out", 0.2)
            while not done:
                mask = info.get("mask", np.ones(ACTION_SPACE_SIZE, dtype=bool))
                action = act_with_info(agent, obs, mask, info)
                obs, reward, term, trunc, info = env.step(action)
                done = term or trunc
                ep_ret += reward
                W += reward
            returns.append(ep_ret)
            if W < stop_out_frac * env.W0:
                ruin += 1
        returns_arr = np.array(returns)
        metrics = {
            "agent": name,
            "return_mean": float(returns_arr.mean()),
            "return_std": float(returns_arr.std()),
            "sharpe": float(returns_arr.mean() / (returns_arr.std() + 1e-8)),
            "ruin_prob": float(ruin / n_episodes),
        }
        results.append(metrics)

    return pd.DataFrame(results)


# -------- Two-player evaluation (IPPO/MAPPO vs baselines) --------

def run_matchup_two_player(rl_agent, agent_type: str, opponent, cfg: Dict[str, Any], n_episodes: int, seed: int):
    env = TwoPlayerCardEnv(cfg)
    rng = np.random.RandomState(seed)
    returns_a: List[float] = []
    returns_b: List[float] = []

    is_level1 = isinstance(opponent, Level1Policy)

    for _ in range(n_episodes):
        (obs_a, obs_b), info = env.reset(seed=int(rng.randint(0, 1e9)))
        ep_ret_a = 0.0
        ep_ret_b = 0.0
        done = False
        if is_level1:
            opponent.opp_history.clear()

        while not done:
            mask_a = info["mask_a"]
            mask_b = info["mask_b"]

            if agent_type == "mappo":
                joint_obs = np.concatenate([obs_a, obs_b])
                act_a, _, _ = rl_agent.step(obs_a, joint_obs, mask_a)
            else:  # ippo
                act_a, _, _ = rl_agent.step(obs_a, mask_a)

            if isinstance(opponent, (EVOracleAgent, EVRealisticAgent, Level1Policy, Level1RealisticPolicy)):
                act_b = opponent.act(obs_b, mask_b, info=info)
            else:
                act_b = opponent.act(obs_b, mask_b)

            (obs_a, obs_b), (r_a, r_b), term, trunc, info = env.step((act_a, act_b))
            done = term or trunc
            ep_ret_a += r_a
            ep_ret_b += r_b

            if is_level1:
                side_a, size_a = decode_action(act_a)
                opponent.update(side_a, size_a)

        returns_a.append(ep_ret_a)
        returns_b.append(ep_ret_b)

    ret_a = np.array(returns_a)
    ret_b = np.array(returns_b)
    return {
        "mean_return_rl": float(ret_a.mean()),
        "std_return_rl": float(ret_a.std()),
        "mean_return_opp": float(ret_b.mean()),
        "std_return_opp": float(ret_b.std()),
    }


def evaluate_two_player(agent_type: str, rl_agent, cfg: Dict[str, Any], n_episodes: int, seed: int):
    baselines = {
        "random_valid": RandomValidAgent(),
        "ev_oracle": EVOracleAgent(),
        "ev_realistic": EVRealisticAgent(),
        "level1": Level1Policy(),
        "level1_realistic": Level1RealisticPolicy(),
    }
    rows = []
    for name, opp in baselines.items():
        metrics = run_matchup_two_player(rl_agent, agent_type, opp, cfg, n_episodes, seed)
        metrics["opponent"] = name
        rows.append(metrics)
    return pd.DataFrame(rows)


# -------- RL vs RL evaluation (self-play & cross-play) --------

def run_rl_vs_rl_matchup(
    agent_a, agent_a_type: str,
    agent_b, agent_b_type: str,
    cfg: Dict[str, Any],
    n_episodes: int,
    seed: int
):
    """
    Run matchup between two RL agents.
    
    Args:
        agent_a: First RL agent (plays as player A)
        agent_a_type: Type string ("ippo", "mappo")
        agent_b: Second RL agent (plays as player B)
        agent_b_type: Type string ("ippo", "mappo")
        cfg: Environment config
        n_episodes: Number of episodes
        seed: Random seed
    
    Returns:
        Dict with performance metrics for both agents
    """
    env = TwoPlayerCardEnv(cfg)
    rng = np.random.RandomState(seed)
    returns_a: List[float] = []
    returns_b: List[float] = []

    for _ in range(n_episodes):
        (obs_a, obs_b), info = env.reset(seed=int(rng.randint(0, 1e9)))
        ep_ret_a = 0.0
        ep_ret_b = 0.0
        done = False

        while not done:
            mask_a = info["mask_a"]
            mask_b = info["mask_b"]

            # Agent A acts
            if agent_a_type == "mappo":
                joint_obs = np.concatenate([obs_a, obs_b])
                act_a, _, _ = agent_a.step(obs_a, joint_obs, mask_a)
            else:  # ippo
                act_a, _, _ = agent_a.step(obs_a, mask_a)

            # Agent B acts
            if agent_b_type == "mappo":
                joint_obs = np.concatenate([obs_a, obs_b])
                act_b, _, _ = agent_b.step(obs_b, joint_obs, mask_b)
            else:  # ippo
                act_b, _, _ = agent_b.step(obs_b, mask_b)

            (obs_a, obs_b), (r_a, r_b), term, trunc, info = env.step((act_a, act_b))
            done = term or trunc
            ep_ret_a += r_a
            ep_ret_b += r_b

        returns_a.append(ep_ret_a)
        returns_b.append(ep_ret_b)

    ret_a = np.array(returns_a)
    ret_b = np.array(returns_b)
    return {
        "mean_return_a": float(ret_a.mean()),
        "std_return_a": float(ret_a.std()),
        "sharpe_a": float(ret_a.mean() / (ret_a.std() + 1e-8)),
        "mean_return_b": float(ret_b.mean()),
        "std_return_b": float(ret_b.std()),
        "sharpe_b": float(ret_b.mean() / (ret_b.std() + 1e-8)),
    }


def evaluate_rl_vs_rl(
    agents_dict: Dict[str, tuple],  # {name: (agent, agent_type)}
    cfg: Dict[str, Any],
    n_episodes: int,
    seed: int,
    include_self_play: bool = True
):
    """
    Evaluate all RL agents against each other.
    
    Args:
        agents_dict: Dict mapping agent name to (agent_obj, agent_type)
        cfg: Environment config
        n_episodes: Number of episodes per matchup
        seed: Random seed
        include_self_play: Whether to include self-play matchups
    
    Returns:
        DataFrame with results for all matchups
    """
    rows = []
    agent_names = list(agents_dict.keys())
    
    for i, name_a in enumerate(agent_names):
        agent_a, type_a = agents_dict[name_a]
        
        # Determine which agents to play against
        if include_self_play:
            opponents = agent_names[i:]  # Include self
        else:
            opponents = agent_names[i+1:]  # Exclude self
        
        for name_b in opponents:
            agent_b, type_b = agents_dict[name_b]
            
            print(f"Running: {name_a} vs {name_b}...")
            metrics = run_rl_vs_rl_matchup(
                agent_a, type_a,
                agent_b, type_b,
                cfg, n_episodes, seed
            )
            
            metrics["agent_a"] = name_a
            metrics["agent_b"] = name_b
            rows.append(metrics)
            
            # If not self-play, also run reverse matchup (B vs A)
            if name_a != name_b:
                print(f"Running: {name_b} vs {name_a}...")
                metrics_reverse = run_rl_vs_rl_matchup(
                    agent_b, type_b,
                    agent_a, type_a,
                    cfg, n_episodes, seed
                )
                metrics_reverse["agent_a"] = name_b
                metrics_reverse["agent_b"] = name_a
                rows.append(metrics_reverse)
    
    return pd.DataFrame(rows)


# -------- Main CLI --------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["dqn", "ippo", "mappo"], required=True)
    parser.add_argument("--model_path", type=str, default=None, help="Path to .pt model")
    parser.add_argument("--opponent", choices=["baselines", "ippo", "mappo"], default="baselines",
                       help="Opponent type: 'baselines' (default) for rule-based, or 'ippo'/'mappo' for RL agent")
    parser.add_argument("--opponent_path", type=str, default=None, help="Path to opponent RL model (if opponent is ippo/mappo)")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None, help="Optional CSV output path")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Defaults matching training
    single_cfg = {
        "W0": 500.0,
        "episode_length": 20,
        "flags": {"enable_events": True, "enable_impact": False},
        "pass_penalty": 0.05
    }
    two_cfg = {
        "episode_length": 20,
        "W0": 500.0,
        "flags": {"enable_events": True, "enable_impact": False},
        "pass_penalty": 0.05
    }

    # Load RL agent
    if args.model_path is None:
        default_path = os.path.join("data", "models", args.agent, f"{args.agent}_final.pt")
    else:
        default_path = args.model_path

    if args.agent == "dqn":
        rl_agent = load_dqn(default_path, device)
        agents = {
            "dqn": rl_agent,
            "random_valid": RandomValidAgent(),
            "ev_oracle": EVOracleAgent(),
            "ev_realistic": EVRealisticAgent(),
            "level1": Level1Policy(),
            "level1_realistic": Level1RealisticPolicy(),
        }
        df = evaluate_single_agents(agents, single_cfg, args.episodes, args.seed)
    elif args.agent in ["ippo", "mappo"]:
        # Load main agent
        if args.agent == "ippo":
            rl_agent = load_ippo(default_path, device)
        else:  # mappo
            rl_agent = load_mappo(default_path, device)
        
        # Check opponent type
        if args.opponent == "baselines":
            # Original behavior: test against baselines
            df = evaluate_two_player(args.agent, rl_agent, two_cfg, args.episodes, args.seed)
        else:
            # New: test against another RL agent
            if args.opponent_path is None:
                opp_path = os.path.join("data", "models", args.opponent, f"{args.opponent}_final.pt")
            else:
                opp_path = args.opponent_path
            
            # Load opponent
            if args.opponent == "ippo":
                opp_agent = load_ippo(opp_path, device)
            else:  # mappo
                opp_agent = load_mappo(opp_path, device)
            
            # Run RL vs RL matchup
            print(f"Evaluating {args.agent.upper()} vs {args.opponent.upper()}...")
            metrics = run_rl_vs_rl_matchup(
                rl_agent, args.agent,
                opp_agent, args.opponent,
                two_cfg, args.episodes, args.seed
            )
            metrics["agent_a"] = args.agent.upper()
            metrics["agent_b"] = args.opponent.upper()
            df = pd.DataFrame([metrics])

    print(df)
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
