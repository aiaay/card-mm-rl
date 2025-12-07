import argparse
import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from mmrl.env.single_env import SingleCardEnv
from mmrl.agents.dqn.dqn_agent import DQNAgent
from mmrl.env.spaces import get_obs_shape, ACTION_SPACE_SIZE

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", type=str, default="data/logs/dqn")
    parser.add_argument("--checkpoint-dir", type=str, default="data/models/dqn")
    args = parser.parse_args()
    
    # Config
    cfg = {
        # Longer horizon to allow more actions per episode
        "episode_length": 20,
        "W0": 500.0,
        # Start with impact off during exploration
        "flags": {"enable_events": True, "enable_impact": False},
        # Softer stop-out threshold
        "stop_out": 0.1,
        # Incentive to trade
        "pass_penalty": 0.05
    }
    
    env = SingleCardEnv(cfg)
    
    agent_cfg = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gamma": 0.99,
        "batch_size": 64,
        "lr": 1e-3,
        "epsilon_start": 1.0,
        "epsilon_min": 0.1,
        # Slower decay to keep exploring longer
        "epsilon_decay": 0.999,
        "buffer_size": 20000,
        "hidden_dims": [64, 64],
        "target_update_freq": 200,
    }
    
    agent = DQNAgent(get_obs_shape()[0], ACTION_SPACE_SIZE, agent_cfg)
    
    # Setup logging
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(args.log_dir)
    
    returns = []
    
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        done = False
        ep_ret = 0.0
        ep_len = 0
        
        while not done:
            mask = info["mask"]
            action = agent.act(obs, mask)
            
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated
            
            next_mask = next_info.get("mask")
            if next_mask is None:
                next_mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
            
            agent.step(obs, action, reward, next_obs, done, next_mask)
            
            obs = next_obs
            info = next_info
            ep_ret += reward
            ep_len += 1
            
        returns.append(ep_ret)
        
        # Log to TensorBoard
        writer.add_scalar("train/episode_return", ep_ret, ep)
        writer.add_scalar("train/episode_length", ep_len, ep)
        writer.add_scalar("train/epsilon", agent.epsilon, ep)
        
        if (ep + 1) % 100 == 0:
            avg_return = np.mean(returns[-100:])
            writer.add_scalar("train/avg_return_100", avg_return, ep)
            print(f"Episode {ep+1}/{args.episodes} | Return: {ep_ret:.2f} | Avg100: {avg_return:.2f} | Epsilon: {agent.epsilon:.3f}")
            
        # Save checkpoint
        if (ep + 1) % 500 == 0 or ep == args.episodes - 1:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"dqn_ep{ep+1}.pt")
            torch.save({
                'q_net': agent.q_net.state_dict(),
                'target_net': agent.target_net.state_dict(),
                'optimizer': agent.optimizer.state_dict(),
                'episode': ep + 1,
                'epsilon': agent.epsilon
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
    
    writer.close()
    print(f"DQN Training finished. Logs: {args.log_dir}, Models: {args.checkpoint_dir}")

    # Save trained Q-network
    save_dir = os.path.join("data", "models", "dqn")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "dqn_final.pt")
    torch.save(agent.q_net.state_dict(), save_path)
    print(f"Saved DQN model to {save_path}")

if __name__ == "__main__":
    main()
