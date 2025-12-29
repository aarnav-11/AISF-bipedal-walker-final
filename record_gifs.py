#!/usr/bin/env python3
"""
Record GIFs of the trained walker, keeping only successful runs.

Usage:
    python record_gifs.py <checkpoint_path> [options]
    
Examples:
    python record_gifs.py results/run_xxx/checkpoint_best.pkl --episodes 20 --min-reward 300
"""

import os
import pickle
import shutil
import numpy as np
from pathlib import Path

import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from src.network import BipedalController


def record_episodes(
    checkpoint_path: str,
    output_dir: str = "gifs",
    n_episodes: int = 20,
    min_reward: float = 300.0
):
    """
    Record episodes as videos, delete those below min_reward threshold.
    
    Args:
        checkpoint_path: Path to trained checkpoint
        output_dir: Directory to save videos
        n_episodes: Number of episodes to attempt
        min_reward: Minimum reward to keep the recording
    """
    # Load checkpoint
    print(f"📂 Loading checkpoint: {checkpoint_path}")
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    # Create controller and restore state
    controller = BipedalController()
    controller.set_state(checkpoint['controller_state'])
    print("✅ Controller loaded successfully")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    kept_count = 0
    rewards = []
    
    print(f"\n🎬 Recording {n_episodes} episodes (keeping reward >= {min_reward})...\n")
    
    for ep in range(n_episodes):
        # Create environment with video recording
        video_folder = output_path / f"temp_ep{ep}"
        env = gym.make("BipedalWalker-v3", render_mode="rgb_array")
        env = RecordVideo(
            env, 
            video_folder=str(video_folder),
            name_prefix=f"walker",
            episode_trigger=lambda x: True  # Record every episode
        )
        
        obs, _ = env.reset()
        controller.soft_reset()
        
        episode_reward = 0.0
        done = False
        steps = 0
        
        while not done:
            action = controller.step(obs, 0.0)  # No learning during eval
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            steps += 1
        
        env.close()
        rewards.append(episode_reward)
        
        # Find the recorded video file
        video_files = list(video_folder.glob("*.mp4"))
        
        if episode_reward >= min_reward:
            # Keep it - move to main folder with reward in name
            for vf in video_files:
                new_name = output_path / f"walker_reward{episode_reward:.0f}_ep{ep}.mp4"
                vf.rename(new_name)
                print(f"✅ Kept: {new_name.name} (reward: {episode_reward:.1f}, steps: {steps})")
            kept_count += 1
        else:
            # Delete it
            print(f"❌ Deleted: ep{ep} (reward: {episode_reward:.1f} < {min_reward}, steps: {steps})")
        
        # Clean up temp folder
        if video_folder.exists():
            shutil.rmtree(video_folder)
    
    # Summary
    print(f"\n{'='*50}")
    print(f"📊 Summary")
    print(f"{'='*50}")
    print(f"Episodes recorded: {n_episodes}")
    print(f"Episodes kept: {kept_count} ({100*kept_count/n_episodes:.1f}%)")
    print(f"Mean reward: {np.mean(rewards):.1f}")
    print(f"Best reward: {np.max(rewards):.1f}")
    print(f"Worst reward: {np.min(rewards):.1f}")
    print(f"📁 Videos saved to: {output_path.absolute()}")
    
    if kept_count == 0:
        print(f"\n⚠️  No episodes reached {min_reward} reward.")
        print(f"   Try lowering --min-reward or training longer.")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Record GIFs of trained walker, keeping only successful runs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "checkpoint", 
        type=str,
        help="Path to checkpoint file"
    )
    parser.add_argument(
        "--episodes", 
        type=int, 
        default=20, 
        help="Number of episodes to record"
    )
    parser.add_argument(
        "--min-reward", 
        type=float, 
        default=300.0, 
        help="Minimum reward threshold to keep recording"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="gifs", 
        help="Output directory for videos"
    )
    
    args = parser.parse_args()
    
    record_episodes(
        args.checkpoint, 
        args.output, 
        args.episodes, 
        args.min_reward
    )


if __name__ == "__main__":
    main()

