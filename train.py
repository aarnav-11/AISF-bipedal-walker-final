#!/usr/bin/env python3
"""
Training script for the Dynamic Synapse Bipedal Walker Controller.

This script trains the bio-inspired neural network to control the BipedalWalker-v3
environment using reward-modulated learning with oscillating synaptic weights.

Usage:
    python train.py [options]
    
Options:
    --episodes: Number of episodes to train (default: 10000)
    --render: Enable rendering during training
    --repulsive: Enable repulsive learning (default: True)
    --adaptation: Adaptation type: none, linear, nonlinear (default: nonlinear)
    --seed: Random seed for reproducibility
    --save-every: Save checkpoint every N episodes (default: 100)
"""

import argparse
import os
import time
import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Tuple
from collections import deque

import numpy as np

try:
    import gymnasium as gym
except ImportError:
    import gym

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel
from rich.live import Live

from src.network import BipedalController

console = Console()


@dataclass
class TrainingConfig:
    """Training configuration."""
    n_episodes: int = 10000
    max_steps: int = 1600  # Max steps per episode
    render: bool = False
    repulsive_learning: bool = True
    adaptation: str = "nonlinear"  # none, linear, nonlinear
    seed: int = 42
    save_every: int = 100
    output_dir: str = "results"
    dt: float = 33.0  # Time step in ms
    
    # Tracking
    solved_threshold: float = 300.0
    solved_window: int = 100


class TrainingStats:
    """Track training statistics."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.episode_rewards: deque = deque(maxlen=window_size)
        self.episode_lengths: deque = deque(maxlen=window_size)
        self.best_reward: float = float('-inf')
        self.total_steps: int = 0
        self.start_time: float = time.time()
        self.solved_episode: Optional[int] = None
    
    def add_episode(self, reward: float, length: int):
        """Record episode statistics."""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.total_steps += length
        
        if reward > self.best_reward:
            self.best_reward = reward
    
    @property
    def mean_reward(self) -> float:
        """Mean reward over recent episodes."""
        if not self.episode_rewards:
            return 0.0
        return np.mean(self.episode_rewards)
    
    @property
    def mean_length(self) -> float:
        """Mean episode length over recent episodes."""
        if not self.episode_lengths:
            return 0.0
        return np.mean(self.episode_lengths)
    
    @property
    def elapsed_time(self) -> float:
        """Elapsed training time in seconds."""
        return time.time() - self.start_time
    
    def is_solved(self, threshold: float = 300.0) -> bool:
        """Check if environment is solved (mean reward >= threshold)."""
        return len(self.episode_rewards) >= self.window_size and self.mean_reward >= threshold


def create_output_dir(config: TrainingConfig) -> Path:
    """Create output directory for this run."""
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"run_{timestamp}_seed{config.seed}"
    
    if config.repulsive_learning:
        run_name += "_repulsive"
    run_name += f"_{config.adaptation}"
    
    output_path = Path(config.output_dir) / run_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_path / "config.txt", "w") as f:
        for key, value in vars(config).items():
            f.write(f"{key}: {value}\n")
    
    return output_path


def save_checkpoint(
    controller: BipedalController,
    stats: TrainingStats,
    episode: int,
    output_path: Path,
    is_best: bool = False
):
    """Save training checkpoint."""
    checkpoint = {
        'episode': episode,
        'controller_state': controller.get_state(),
        'stats': {
            'episode_rewards': list(stats.episode_rewards),
            'episode_lengths': list(stats.episode_lengths),
            'best_reward': stats.best_reward,
            'total_steps': stats.total_steps,
            'solved_episode': stats.solved_episode,
        }
    }
    
    # Save latest
    with open(output_path / "checkpoint_latest.pkl", "wb") as f:
        pickle.dump(checkpoint, f)
    
    # Save best if applicable
    if is_best:
        with open(output_path / "checkpoint_best.pkl", "wb") as f:
            pickle.dump(checkpoint, f)
    
    # Periodic saves
    if episode % 1000 == 0:
        with open(output_path / f"checkpoint_ep{episode}.pkl", "wb") as f:
            pickle.dump(checkpoint, f)


def load_checkpoint(path: Path, controller: BipedalController) -> Tuple[int, TrainingStats]:
    """Load training checkpoint."""
    with open(path, "rb") as f:
        checkpoint = pickle.load(f)
    
    controller.set_state(checkpoint['controller_state'])
    
    stats = TrainingStats()
    stats.episode_rewards = deque(checkpoint['stats']['episode_rewards'], maxlen=100)
    stats.episode_lengths = deque(checkpoint['stats']['episode_lengths'], maxlen=100)
    stats.best_reward = checkpoint['stats']['best_reward']
    stats.total_steps = checkpoint['stats']['total_steps']
    stats.solved_episode = checkpoint['stats']['solved_episode']
    
    return checkpoint['episode'], stats


def create_status_table(
    episode: int,
    stats: TrainingStats,
    config: TrainingConfig,
    controller: BipedalController
) -> Table:
    """Create a rich table showing training status."""
    table = Table(title=f"Training Progress - Episode {episode}", expand=True)
    
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Mean Reward (100 ep)", f"{stats.mean_reward:.2f}")
    table.add_row("Best Reward", f"{stats.best_reward:.2f}")
    table.add_row("Mean Episode Length", f"{stats.mean_length:.1f}")
    table.add_row("Total Steps", f"{stats.total_steps:,}")
    table.add_row("Elapsed Time", f"{stats.elapsed_time / 60:.1f} min")
    
    if stats.solved_episode is not None:
        table.add_row("Solved at Episode", f"{stats.solved_episode}", style="bold green")
    
    # Layer 1 amplitude (exploration indicator)
    amp = controller.layer1._amplitudes.mean()
    table.add_row("Layer 1 Amplitude (avg)", f"{amp:.4f}")
    
    # Neuron sensitivity
    sens = controller.layer2_neurons.sensitivity.mean()
    table.add_row("Neuron Sensitivity (avg)", f"{sens:.4f}")
    
    return table


def train(config: TrainingConfig):
    """Main training loop."""
    console.print(Panel.fit(
        "[bold cyan]Dynamic Synapse Bipedal Walker Training[/bold cyan]\n"
        "[dim]Bio-inspired reinforcement learning with oscillating synapses[/dim]",
        border_style="blue"
    ))
    
    # Set random seed
    np.random.seed(config.seed)
    
    # Create environment
    try:
        env = gym.make("BipedalWalker-v3", render_mode="human" if config.render else None)
    except:
        # Fallback for older gym versions
        env = gym.make("BipedalWalker-v3")
    
    # Create controller
    controller = BipedalController(
        dt=config.dt,
        repulsive_learning=config.repulsive_learning,
        use_nonlinear_adaptation=(config.adaptation == "nonlinear"),
    )
    
    # Create output directory
    output_path = create_output_dir(config)
    console.print(f"[dim]Saving results to: {output_path}[/dim]")
    
    # Training stats
    stats = TrainingStats(window_size=config.solved_window)
    
    # Training loop
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        
        task = progress.add_task("[cyan]Training...", total=config.n_episodes)
        
        for episode in range(config.n_episodes):
            # Reset environment only - DON'T reset controller weights!
            # The learning accumulates across episodes
            obs, info = env.reset(seed=config.seed + episode)
            
            # Only reset CPG state and action buffer, NOT learned weights
            controller.cpg.reset()
            controller.last_action = np.zeros(controller.n_actions)
            
            episode_reward = 0.0
            episode_steps = 0
            step_reward = 0.0  # Reward from previous step
            
            for step in range(config.max_steps):
                # Get action from controller using STEP reward (not cumulative)
                action = controller.step(obs, step_reward)
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                step_reward = reward  # Save for next iteration
                episode_reward += reward
                episode_steps += 1
                
                if done:
                    break
            
            # Record stats
            stats.add_episode(episode_reward, episode_steps)
            
            # Check if solved
            if stats.is_solved(config.solved_threshold) and stats.solved_episode is None:
                stats.solved_episode = episode
                console.print(f"\n[bold green]🎉 Solved at episode {episode}![/bold green]")
            
            # Update progress
            progress.update(
                task,
                advance=1,
                description=f"[cyan]Ep {episode}: R={episode_reward:.1f}, Avg={stats.mean_reward:.1f}"
            )
            
            # Save checkpoint
            is_best = episode_reward >= stats.best_reward
            if episode % config.save_every == 0 or is_best:
                save_checkpoint(controller, stats, episode, output_path, is_best)
            
            # Print periodic status
            if episode % 100 == 0:
                table = create_status_table(episode, stats, config, controller)
                console.print(table)
    
    # Final save
    save_checkpoint(controller, stats, config.n_episodes, output_path, False)
    
    # Final summary
    console.print("\n")
    console.print(Panel.fit(
        f"[bold]Training Complete![/bold]\n\n"
        f"Episodes: {config.n_episodes}\n"
        f"Best Reward: {stats.best_reward:.2f}\n"
        f"Final Mean Reward: {stats.mean_reward:.2f}\n"
        f"Total Steps: {stats.total_steps:,}\n"
        f"Training Time: {stats.elapsed_time / 60:.1f} minutes\n"
        f"Solved: {'Yes at episode ' + str(stats.solved_episode) if stats.solved_episode else 'No'}",
        border_style="green" if stats.solved_episode else "yellow"
    ))
    
    env.close()
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Train Dynamic Synapse Bipedal Walker Controller",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--episodes", type=int, default=10000,
        help="Number of training episodes"
    )
    parser.add_argument(
        "--render", action="store_true",
        help="Enable rendering during training"
    )
    parser.add_argument(
        "--repulsive", type=lambda x: x.lower() == 'true', default=True,
        help="Enable repulsive learning (True/False)"
    )
    parser.add_argument(
        "--adaptation", choices=["none", "linear", "nonlinear"], default="nonlinear",
        help="Adaptation type for neuron sensitivity"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--save-every", type=int, default=100,
        help="Save checkpoint every N episodes"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    config = TrainingConfig(
        n_episodes=args.episodes,
        render=args.render,
        repulsive_learning=args.repulsive,
        adaptation=args.adaptation,
        seed=args.seed,
        save_every=args.save_every,
        output_dir=args.output_dir,
    )
    
    train(config)


if __name__ == "__main__":
    main()

