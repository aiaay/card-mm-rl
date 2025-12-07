import argparse
import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from mmrl.env.two_player_env import TwoPlayerCardEnv
from mmrl.agents.mappo.mappo_agent import MAPPOAgent
from mmrl.env.spaces import get_obs_shape, ACTION_SPACE_SIZE

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", type=str, default="data/logs/mappo")
    parser.add_argument("--checkpoint-dir", type=str, default="data/models/mappo")
    args = parser.parse_args()
    
    cfg = {
        # Longer horizon to encourage more trades per episode
        "episode_length": 20,
        "W0": 500.0,
        # Start without market impact while exploring
        "flags": {"enable_events": True, "enable_impact": False},
        # Softer stop-out
        "stop_out": 0.1,
        # Incentive to trade
        "pass_penalty": 0.05
    }
    
    env = TwoPlayerCardEnv(cfg)
    obs_dim = get_obs_shape()[0]
    act_dim = ACTION_SPACE_SIZE
    joint_obs_dim = obs_dim * 2
    
    agent_cfg = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "lr": 3e-4,
        "rollout_steps": 1024,
        "train_iters": 6,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        # Encourage exploration
        "entropy_coef": 0.02,
        "clip_ratio": 0.2,
        "value_coef": 0.5,
    }
    
    # Shared policy, centralized critic
    agent_a = MAPPOAgent(obs_dim, joint_obs_dim, act_dim, agent_cfg)
    agent_b = MAPPOAgent(obs_dim, joint_obs_dim, act_dim, agent_cfg)
    agent_b.ac = agent_a.ac
    agent_b.optimizer = agent_a.optimizer
    
    # Setup logging
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(args.log_dir)
    
    (obs_a, obs_b), info = env.reset(seed=args.seed)
    
    total_steps = 0
    episode_count = 0
    episode_returns = []
    ep_ret_a = 0.0
    ep_ret_b = 0.0
    
    while total_steps < args.steps:
        mask_a = info["mask_a"]
        mask_b = info["mask_b"]
        
        # Joint observation
        joint_obs = np.concatenate([obs_a, obs_b])
        
        act_a, logp_a, val_a = agent_a.step(obs_a, joint_obs, mask_a)
        act_b, logp_b, val_b = agent_b.step(obs_b, joint_obs, mask_b)
        
        (next_obs_a, next_obs_b), (r_a, r_b), term, trunc, next_info = env.step((act_a, act_b))
        done = term or trunc
        
        agent_a.store(obs_a, joint_obs, act_a, r_a, logp_a, val_a, done, mask_a)
        agent_b.store(obs_b, joint_obs, act_b, r_b, logp_b, val_b, done, mask_b)
        
        obs_a, obs_b = next_obs_a, next_obs_b
        info = next_info
        total_steps += 1
        ep_ret_a += r_a
        ep_ret_b += r_b
        
        if done:
            # Bootstrap
            next_joint_obs = np.concatenate([next_obs_a, next_obs_b])
            _, _, last_val_a = agent_a.step(obs_a, next_joint_obs, next_info["mask_a"])
            _, _, last_val_b = agent_b.step(obs_b, next_joint_obs, next_info["mask_b"])
            
            # Update
            if agent_a.buffer.ptr >= 64:
                loss_a = agent_a.update(last_val_a)
                loss_b = agent_b.update(last_val_b)
                
                if loss_a is not None:
                    writer.add_scalar("train/loss_a", loss_a, episode_count)
                if loss_b is not None:
                    writer.add_scalar("train/loss_b", loss_b, episode_count)
            
            # Log episode
            writer.add_scalar("train/episode_return_a", ep_ret_a, episode_count)
            writer.add_scalar("train/episode_return_b", ep_ret_b, episode_count)
            episode_returns.append((ep_ret_a, ep_ret_b))
            
            episode_count += 1
            
            if episode_count % 100 == 0:
                avg_a = np.mean([r[0] for r in episode_returns[-100:]])
                avg_b = np.mean([r[1] for r in episode_returns[-100:]])
                print(f"Episode {episode_count} | Steps {total_steps}/{args.steps} | Avg100 A: {avg_a:.2f} B: {avg_b:.2f}")
            
            # Reset
            (obs_a, obs_b), info = env.reset()
            ep_ret_a = 0.0
            ep_ret_b = 0.0
            
        # Save checkpoint periodically
        if total_steps % 10000 == 0 and total_steps > 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"mappo_step{total_steps}.pt")
            torch.save({
                'ac': agent_a.ac.state_dict(),
                'optimizer': agent_a.optimizer.state_dict(),
                'total_steps': total_steps,
                'episode_count': episode_count
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
    
    # Final checkpoint
    final_path = os.path.join(args.checkpoint_dir, "mappo_final.pt")
    torch.save({
        'ac': agent_a.ac.state_dict(),
        'optimizer': agent_a.optimizer.state_dict(),
        'total_steps': total_steps,
        'episode_count': episode_count
    }, final_path)
    
    writer.close()
    print(f"MAPPO Training finished. Logs: {args.log_dir}, Models: {args.checkpoint_dir}")

    # Save shared actor-critic weights
    save_dir = os.path.join("data", "models", "mappo")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "mappo_final.pt")
    torch.save(agent_a.ac.state_dict(), save_path)
    print(f"Saved MAPPO model to {save_path}")

if __name__ == "__main__":
    main()
