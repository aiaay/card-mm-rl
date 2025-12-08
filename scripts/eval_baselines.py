#!/usr/bin/env python
"""
Evaluate all baseline policies and print results.
"""
import argparse
import numpy as np
import pandas as pd
import os

from mmrl.env.single_env import SingleCardEnv
from mmrl.baselines.random_valid import RandomValidAgent
from mmrl.baselines.ev_oracle import EVOracleAgent
from mmrl.baselines.ev_realistic import EVRealisticAgent
from mmrl.baselines.level1_crowding import Level1Policy
from mmrl.baselines.level1_realistic import Level1RealisticPolicy

def evaluate_baseline(agent, agent_name: str, cfg: dict, n_episodes: int = 100, seed: int = 42):
    """
    Evaluate a baseline agent that may need info dict.
    """
    env = SingleCardEnv(cfg)
    
    returns = []
    valid_actions = 0
    total_actions = 0
    
    for i in range(n_episodes):
        obs, info = env.reset(seed=seed + i)
        done = False
        ep_ret = 0.0
        
        while not done:
            mask = info["mask"]
            
            # Call agent.act with info if it accepts it
            try:
                # Try with info first (for EV Oracle, Level-1)
                action = agent.act(obs, mask, info=info, eval_mode=True)
            except TypeError:
                # Fallback for agents that don't accept info (Random)
                try:
                    action = agent.act(obs, mask, eval_mode=True)
                except TypeError:
                    # Fallback for Level1Policy which doesn't have eval_mode
                    action = agent.act(obs, mask, info)
            
            if mask[action]:
                valid_actions += 1
            total_actions += 1
            
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            ep_ret += reward
            
        returns.append(ep_ret)
    
    # Compute metrics
    returns = np.array(returns)
    return {
        "agent": agent_name,
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "sharpe": float(np.mean(returns) / (np.std(returns) + 1e-8)),
        "valid_rate": float(valid_actions / total_actions) if total_actions > 0 else 0.0,
        "n_episodes": n_episodes
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-episodes", type=int, default=100)
    parser.add_argument("--output", type=str, default="data/results/baselines.csv")
    args = parser.parse_args()
    
    cfg = {
        "episode_length": 10,
        "W0": 500.0,
        "flags": {"enable_events": True, "enable_impact": True}
    }
    
    results = []
    
    print("="*50)
    print("Evaluating Baselines")
    print("="*50)
    
    # 1. Random-Valid
    print("\n1. Random-Valid...")
    random_agent = RandomValidAgent()
    results.append(evaluate_baseline(random_agent, "Random", cfg, args.n_episodes))
    
    # 2. EV Oracle (privileged - knows true mu)
    print("2. EV Oracle (Level-0, privileged)...")
    oracle_agent = EVOracleAgent()
    results.append(evaluate_baseline(oracle_agent, "EV_Oracle", cfg, args.n_episodes))
    
    # 3. EV Realistic (no privileged info - computes fair value from obs only)
    print("3. EV Realistic (no privileged info)...")
    realistic_agent = EVRealisticAgent(alpha=0.15)
    results.append(evaluate_baseline(realistic_agent, "EV_Realistic", cfg, args.n_episodes))
    
    # 4. Level-1 Crowding (privileged - uses info["mu"])
    print("4. Level-1 Crowding (privileged)...")
    level1_agent = Level1Policy(history_len=10, alpha=0.3)
    results.append(evaluate_baseline(level1_agent, "Level1", cfg, args.n_episodes))
    
    # 5. Level-1 Realistic (no privileged info)
    print("5. Level-1 Realistic (no privileged info)...")
    level1_realistic_agent = Level1RealisticPolicy(history_len=10, alpha=0.15)
    results.append(evaluate_baseline(level1_realistic_agent, "Level1_Realistic", cfg, args.n_episodes))
    
    # Save results
    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    
    print("\n" + "="*50)
    print("Baseline Evaluation Results")
    print("="*50)
    print(df.to_string(index=False))
    print(f"\nSaved to: {args.output}")

if __name__ == "__main__":
    main()

