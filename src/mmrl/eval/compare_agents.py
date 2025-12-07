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
from mmrl.baselines.level1_crowding import Level1Policy


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

            if isinstance(opponent, (EVOracleAgent, Level1Policy)):
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
        "level1": Level1Policy(),
    }
    rows = []
    for name, opp in baselines.items():
        metrics = run_matchup_two_player(rl_agent, agent_type, opp, cfg, n_episodes, seed)
        metrics["opponent"] = name
        rows.append(metrics)
    return pd.DataFrame(rows)


# -------- Main CLI --------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["dqn", "ippo", "mappo"], required=True)
    parser.add_argument("--model_path", type=str, default=None, help="Path to .pt model")
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
            "level1": Level1Policy(),
        }
        df = evaluate_single_agents(agents, single_cfg, args.episodes, args.seed)
    elif args.agent == "ippo":
        rl_agent = load_ippo(default_path, device)
        df = evaluate_two_player("ippo", rl_agent, two_cfg, args.episodes, args.seed)
    else:  # mappo
        rl_agent = load_mappo(default_path, device)
        df = evaluate_two_player("mappo", rl_agent, two_cfg, args.episodes, args.seed)

    print(df)
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
