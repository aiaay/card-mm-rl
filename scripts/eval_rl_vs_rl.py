#!/usr/bin/env python
"""
Evaluate trained RL agents against each other (self-play and cross-play).
"""
import argparse
import os
import torch
import pandas as pd

from mmrl.eval.compare_agents import (
    load_ippo, load_mappo, evaluate_rl_vs_rl
)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained RL agents against each other"
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        choices=["ippo", "mappo"],
        default=["ippo", "mappo"],
        help="Which agents to evaluate (default: both)"
    )
    parser.add_argument(
        "--ippo-path",
        type=str,
        default="data/models/ippo/ippo_final.pt",
        help="Path to IPPO checkpoint"
    )
    parser.add_argument(
        "--mappo-path",
        type=str,
        default="data/models/mappo/mappo_final.pt",
        help="Path to MAPPO checkpoint"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of episodes per matchup"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/results/rl_vs_rl.csv",
        help="Output CSV path"
    )
    parser.add_argument(
        "--no-self-play",
        action="store_true",
        help="Skip self-play matchups"
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Environment config (matching training)
    cfg = {
        "episode_length": 20,
        "W0": 500.0,
        "flags": {"enable_events": True, "enable_impact": False},
        "stop_out": 0.1,
        "pass_penalty": 0.05
    }

    # Load agents
    agents_dict = {}
    
    if "ippo" in args.agents:
        if os.path.exists(args.ippo_path):
            print(f"Loading IPPO from {args.ippo_path}...")
            ippo_agent = load_ippo(args.ippo_path, device)
            agents_dict["IPPO"] = (ippo_agent, "ippo")
        else:
            print(f"Warning: IPPO checkpoint not found at {args.ippo_path}")
    
    if "mappo" in args.agents:
        if os.path.exists(args.mappo_path):
            print(f"Loading MAPPO from {args.mappo_path}...")
            mappo_agent = load_mappo(args.mappo_path, device)
            agents_dict["MAPPO"] = (mappo_agent, "mappo")
        else:
            print(f"Warning: MAPPO checkpoint not found at {args.mappo_path}")

    if not agents_dict:
        print("Error: No agents loaded. Check model paths.")
        return

    # Run evaluation
    print("\n" + "="*60)
    print("Evaluating RL Agents Against Each Other")
    print("="*60)
    print(f"Agents: {list(agents_dict.keys())}")
    print(f"Episodes per matchup: {args.episodes}")
    print(f"Self-play: {not args.no_self_play}")
    print("="*60 + "\n")

    df = evaluate_rl_vs_rl(
        agents_dict,
        cfg,
        args.episodes,
        args.seed,
        include_self_play=not args.no_self_play
    )

    # Print results
    print("\n" + "="*60)
    print("Results:")
    print("="*60)
    print(df.to_string(index=False))
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nResults saved to: {args.output}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("Summary Statistics:")
    print("="*60)
    for idx, row in df.iterrows():
        agent_a = row["agent_a"]
        agent_b = row["agent_b"]
        mean_a = row["mean_return_a"]
        mean_b = row["mean_return_b"]
        sharpe_a = row["sharpe_a"]
        sharpe_b = row["sharpe_b"]
        
        print(f"\n{agent_a} (Player A) vs {agent_b} (Player B):")
        print(f"  {agent_a}: Return={mean_a:.2f}, Sharpe={sharpe_a:.2f}")
        print(f"  {agent_b}: Return={mean_b:.2f}, Sharpe={sharpe_b:.2f}")
        
        # Determine winner
        if mean_a > mean_b:
            print(f"  Winner: {agent_a} (+{mean_a - mean_b:.2f})")
        elif mean_b > mean_a:
            print(f"  Winner: {agent_b} (+{mean_b - mean_a:.2f})")
        else:
            print(f"  Result: Tie")


if __name__ == "__main__":
    main()

