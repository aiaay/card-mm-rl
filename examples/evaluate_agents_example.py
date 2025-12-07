#!/usr/bin/env python
"""
Example: How to programmatically evaluate trained agents.

This demonstrates the Python API for running evaluations without using
command-line scripts.
"""

import os
import torch
from mmrl.eval.compare_agents import (
    load_ippo,
    load_mappo,
    evaluate_rl_vs_rl,
    run_rl_vs_rl_matchup,
)


def main():
    """Run example evaluation."""
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Environment configuration
    cfg = {
        "episode_length": 20,
        "W0": 500.0,
        "flags": {"enable_events": True, "enable_impact": False},
        "stop_out": 0.1,
        "pass_penalty": 0.05
    }
    
    # Load trained agents
    print("\nLoading trained agents...")
    
    ippo_path = "data/models/ippo/ippo_final.pt"
    mappo_path = "data/models/mappo/mappo_final.pt"
    
    agents_dict = {}
    
    if os.path.exists(ippo_path):
        ippo = load_ippo(ippo_path, device)
        agents_dict["IPPO"] = (ippo, "ippo")
        print(f"  ✓ Loaded IPPO from {ippo_path}")
    else:
        print(f"  ✗ IPPO not found at {ippo_path}")
    
    if os.path.exists(mappo_path):
        mappo = load_mappo(mappo_path, device)
        agents_dict["MAPPO"] = (mappo, "mappo")
        print(f"  ✓ Loaded MAPPO from {mappo_path}")
    else:
        print(f"  ✗ MAPPO not found at {mappo_path}")
    
    if not agents_dict:
        print("\nError: No agents found. Please train agents first.")
        print("Run: python src/mmrl/agents/ippo/train_ippo.py")
        print("     python src/mmrl/agents/mappo/train_mappo.py")
        return
    
    # Example 1: Single matchup
    print("\n" + "="*60)
    print("Example 1: Single Matchup (IPPO self-play)")
    print("="*60)
    
    if "IPPO" in agents_dict:
        ippo, ippo_type = agents_dict["IPPO"]
        metrics = run_rl_vs_rl_matchup(
            ippo, "ippo",
            ippo, "ippo",
            cfg,
            n_episodes=10,  # Small number for demo
            seed=42
        )
        
        print(f"Player A (IPPO): Return={metrics['mean_return_a']:.2f} ± {metrics['std_return_a']:.2f}")
        print(f"Player B (IPPO): Return={metrics['mean_return_b']:.2f} ± {metrics['std_return_b']:.2f}")
        print(f"Sharpe A: {metrics['sharpe_a']:.3f}, Sharpe B: {metrics['sharpe_b']:.3f}")
    
    # Example 2: All combinations
    if len(agents_dict) > 1:
        print("\n" + "="*60)
        print("Example 2: All Agent Combinations")
        print("="*60)
        
        df = evaluate_rl_vs_rl(
            agents_dict,
            cfg,
            n_episodes=10,  # Small number for demo
            seed=42,
            include_self_play=True
        )
        
        print("\nResults:")
        print(df.to_string(index=False))
        
        # Save results
        output_path = "data/results/example_rl_vs_rl.csv"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
    
    # Example 3: Analysis
    if len(agents_dict) > 1:
        print("\n" + "="*60)
        print("Example 3: Quick Analysis")
        print("="*60)
        
        for idx, row in df.iterrows():
            agent_a = row["agent_a"]
            agent_b = row["agent_b"]
            
            if agent_a == agent_b:
                print(f"\n{agent_a} Self-Play:")
                print(f"  Mean return: {row['mean_return_a']:.2f}")
                print(f"  Balance: {abs(row['mean_return_a'] - row['mean_return_b']):.2f}")
                print(f"  → {'Well balanced' if abs(row['mean_return_a'] - row['mean_return_b']) < 1.0 else 'Asymmetric'}")
            else:
                winner = agent_a if row['mean_return_a'] > row['mean_return_b'] else agent_b
                advantage = abs(row['mean_return_a'] - row['mean_return_b'])
                print(f"\n{agent_a} vs {agent_b}:")
                print(f"  Winner: {winner} (+{advantage:.2f})")


if __name__ == "__main__":
    main()

