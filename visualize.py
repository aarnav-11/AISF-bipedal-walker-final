#!/usr/bin/env python3
"""
Visualization utilities for the Dynamic Synapse Bipedal Walker.

Provides tools for:
- Plotting training progress
- Visualizing network dynamics (weight oscillations, CPG activity)
- Rendering trained policies
- Phase portrait analysis
"""

import argparse
import pickle
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D

try:
    import gymnasium as gym
except ImportError:
    import gym

from src.network import BipedalController
from src.oscillator import FitzHughNagumoOscillator


def set_style():
    """Set matplotlib style for beautiful plots."""
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams.update({
        'figure.figsize': (12, 8),
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'legend.fontsize': 10,
        'figure.dpi': 150,
    })


def plot_training_progress(stats: Dict[str, Any], output_path: Optional[Path] = None):
    """
    Plot training progress curves.
    
    Args:
        stats: Training statistics dictionary
        output_path: Optional path to save the figure
    """
    set_style()
    
    episode_rewards = np.array(stats['episode_rewards'])
    n_episodes = len(episode_rewards)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Episode rewards
    ax = axes[0, 0]
    ax.plot(episode_rewards, alpha=0.5, linewidth=0.5, label='Episode Reward')
    
    # Smoothed reward (moving average)
    window = min(100, n_episodes // 10 + 1)
    if window > 1:
        smoothed = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, n_episodes), smoothed, linewidth=2, label=f'{window}-ep Moving Avg')
    
    ax.axhline(y=300, color='green', linestyle='--', linewidth=1, alpha=0.7, label='Solved Threshold')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Training Reward Progress')
    ax.legend()
    
    # Episode lengths
    ax = axes[0, 1]
    episode_lengths = np.array(stats.get('episode_lengths', []))
    if len(episode_lengths) > 0:
        ax.plot(episode_lengths, alpha=0.5, linewidth=0.5)
        if window > 1 and len(episode_lengths) >= window:
            smoothed_len = np.convolve(episode_lengths, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(episode_lengths)), smoothed_len, linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps')
    ax.set_title('Episode Length')
    
    # Reward distribution
    ax = axes[1, 0]
    ax.hist(episode_rewards, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(x=stats['best_reward'], color='red', linestyle='--', label=f"Best: {stats['best_reward']:.1f}")
    ax.axvline(x=np.mean(episode_rewards), color='blue', linestyle='--', label=f"Mean: {np.mean(episode_rewards):.1f}")
    ax.set_xlabel('Reward')
    ax.set_ylabel('Count')
    ax.set_title('Reward Distribution')
    ax.legend()
    
    # Cumulative progress
    ax = axes[1, 1]
    cumulative = np.cumsum(episode_rewards)
    ax.plot(cumulative)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Cumulative Reward')
    ax.set_title('Cumulative Reward Over Training')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path / 'training_progress.png', dpi=150, bbox_inches='tight')
        plt.savefig(output_path / 'training_progress.pdf', bbox_inches='tight')
    
    plt.show()
    return fig


def plot_network_dynamics(recording: Dict[str, Any], output_path: Optional[Path] = None):
    """
    Plot network dynamics during an episode.
    
    Args:
        recording: Recording dictionary from controller
        output_path: Optional path to save the figure
    """
    set_style()
    
    fig = plt.figure(figsize=(16, 12))
    
    # Layer 1 weight evolution
    ax1 = fig.add_subplot(3, 2, 1)
    layer1_data = recording.get('layer1', {})
    if 'weights' in layer1_data and len(layer1_data['weights']) > 0:
        t = layer1_data['t']
        weights = np.array(layer1_data['weights'])
        # Plot first few weights
        for i in range(min(3, weights.shape[1])):
            ax1.plot(t, weights[:, 0, i], label=f'W[0,{i}]', alpha=0.7)
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Weight')
        ax1.set_title('Layer 1: Dynamic Synapse Weights')
        ax1.legend()
    
    # Layer 1 weight center evolution
    ax2 = fig.add_subplot(3, 2, 2)
    if 'weights_center' in layer1_data and len(layer1_data['weights_center']) > 0:
        centers = np.array(layer1_data['weights_center'])
        for i in range(min(3, centers.shape[1])):
            ax2.plot(t, centers[:, 0, i], label=f'Center[0,{i}]', alpha=0.7)
        ax2.set_xlabel('Time (ms)')
        ax2.set_ylabel('Weight Center')
        ax2.set_title('Layer 1: Learned Weight Centers')
        ax2.legend()
    
    # CPG dynamics
    ax3 = fig.add_subplot(3, 2, 3)
    cpg_data = recording.get('cpg', {})
    if 'V' in cpg_data and len(cpg_data['V']) > 0:
        t = cpg_data['t']
        V = np.array(cpg_data['V'])
        for i in range(V.shape[1]):
            ax3.plot(t, V[:, i], label=f'V[{i}]')
        ax3.set_xlabel('Time (ms)')
        ax3.set_ylabel('Voltage')
        ax3.set_title('CPG: FitzHugh-Nagumo Oscillator Voltages')
        ax3.legend()
    
    # CPG phase portrait
    ax4 = fig.add_subplot(3, 2, 4)
    if 'V' in cpg_data and 'W' in cpg_data:
        V = np.array(cpg_data['V'])
        W = np.array(cpg_data['W'])
        for i in range(V.shape[1]):
            ax4.plot(V[:, i], W[:, i], label=f'Oscillator {i}', alpha=0.7)
        ax4.set_xlabel('V')
        ax4.set_ylabel('W')
        ax4.set_title('CPG: Phase Portrait')
        ax4.legend()
    
    # Reward signal
    ax5 = fig.add_subplot(3, 2, 5)
    if 'reward' in layer1_data:
        ax5.plot(layer1_data['t'], layer1_data['reward'])
        ax5.set_xlabel('Time (ms)')
        ax5.set_ylabel('Reward')
        ax5.set_title('Reward Signal')
    
    # Amplitude decay
    ax6 = fig.add_subplot(3, 2, 6)
    if 'amplitudes' in layer1_data and len(layer1_data['amplitudes']) > 0:
        amps = np.array(layer1_data['amplitudes'])
        ax6.plot(layer1_data['t'], amps.mean(axis=(1, 2)), label='Mean Amplitude')
        ax6.set_xlabel('Time (ms)')
        ax6.set_ylabel('Amplitude')
        ax6.set_title('Layer 1: Oscillation Amplitude (Exploration)')
        ax6.legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path / 'network_dynamics.png', dpi=150, bbox_inches='tight')
    
    plt.show()
    return fig


def plot_weight_space_3d(recording: Dict[str, Any], output_path: Optional[Path] = None):
    """
    3D visualization of weight space trajectory.
    
    Args:
        recording: Recording dictionary from controller
        output_path: Optional path to save the figure
    """
    set_style()
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    layer1_data = recording.get('layer1', {})
    if 'weights' in layer1_data and len(layer1_data['weights']) > 0:
        weights = np.array(layer1_data['weights'])
        
        # Use first 3 weights of first neuron
        if weights.shape[2] >= 3:
            X = weights[:, 0, 0]
            Y = weights[:, 0, 1]
            Z = weights[:, 0, 2]
            
            # Color by time
            colors = np.linspace(0, 1, len(X))
            
            ax.scatter(X, Y, Z, c=colors, cmap='viridis', s=1, alpha=0.5)
            ax.plot(X, Y, Z, alpha=0.3, linewidth=0.5)
            
            # Mark start and end
            ax.scatter([X[0]], [Y[0]], [Z[0]], color='green', s=100, marker='o', label='Start')
            ax.scatter([X[-1]], [Y[-1]], [Z[-1]], color='red', s=100, marker='*', label='End')
            
            ax.set_xlabel('Weight 0')
            ax.set_ylabel('Weight 1')
            ax.set_zlabel('Weight 2')
            ax.set_title('Weight Space Trajectory (Layer 1, Neuron 0)')
            ax.legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path / 'weight_space_3d.png', dpi=150, bbox_inches='tight')
    
    plt.show()
    return fig


def run_evaluation(
    checkpoint_path: Path,
    n_episodes: int = 5,
    render: bool = True,
    record: bool = True
) -> Dict[str, Any]:
    """
    Run evaluation episodes with a trained controller.
    
    Args:
        checkpoint_path: Path to checkpoint file
        n_episodes: Number of episodes to run
        render: Whether to render the environment
        record: Whether to record network dynamics
        
    Returns:
        Dictionary with evaluation results and recordings
    """
    # Load checkpoint
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    # Create controller and restore state
    controller = BipedalController()
    controller.set_state(checkpoint['controller_state'])
    
    # Create environment
    render_mode = "human" if render else None
    try:
        env = gym.make("BipedalWalker-v3", render_mode=render_mode)
    except:
        env = gym.make("BipedalWalker-v3")
    
    results = {
        'episode_rewards': [],
        'episode_lengths': [],
        'recordings': [],
    }
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        controller.reset()
        
        if record:
            controller.start_recording()
        
        episode_reward = 0.0
        steps = 0
        
        while True:
            action = controller.step(obs, episode_reward)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            steps += 1
            
            if render:
                env.render()
            
            if done:
                break
        
        if record:
            controller.stop_recording()
            results['recordings'].append(controller.get_recording())
        
        results['episode_rewards'].append(episode_reward)
        results['episode_lengths'].append(steps)
        
        print(f"Episode {ep + 1}: Reward = {episode_reward:.2f}, Steps = {steps}")
    
    env.close()
    
    print(f"\nMean Reward: {np.mean(results['episode_rewards']):.2f}")
    print(f"Std Reward: {np.std(results['episode_rewards']):.2f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Dynamic Synapse Bipedal Walker",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Plot training progress
    plot_parser = subparsers.add_parser('plot', help='Plot training progress')
    plot_parser.add_argument('checkpoint', type=str, help='Path to checkpoint file')
    plot_parser.add_argument('--output', type=str, default=None, help='Output directory')
    
    # Run evaluation
    eval_parser = subparsers.add_parser('eval', help='Run evaluation episodes')
    eval_parser.add_argument('checkpoint', type=str, help='Path to checkpoint file')
    eval_parser.add_argument('--episodes', type=int, default=5, help='Number of episodes')
    eval_parser.add_argument('--no-render', action='store_true', help='Disable rendering')
    eval_parser.add_argument('--output', type=str, default=None, help='Output directory')
    
    # Plot dynamics
    dynamics_parser = subparsers.add_parser('dynamics', help='Visualize network dynamics')
    dynamics_parser.add_argument('checkpoint', type=str, help='Path to checkpoint file')
    dynamics_parser.add_argument('--output', type=str, default=None, help='Output directory')
    
    args = parser.parse_args()
    
    if args.command == 'plot':
        checkpoint_path = Path(args.checkpoint)
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        output_path = Path(args.output) if args.output else checkpoint_path.parent
        plot_training_progress(checkpoint['stats'], output_path)
    
    elif args.command == 'eval':
        checkpoint_path = Path(args.checkpoint)
        output_path = Path(args.output) if args.output else checkpoint_path.parent
        
        results = run_evaluation(
            checkpoint_path,
            n_episodes=args.episodes,
            render=not args.no_render,
            record=True
        )
        
        # Plot dynamics from first episode
        if results['recordings']:
            plot_network_dynamics(results['recordings'][0], output_path)
            plot_weight_space_3d(results['recordings'][0], output_path)
    
    elif args.command == 'dynamics':
        checkpoint_path = Path(args.checkpoint)
        output_path = Path(args.output) if args.output else checkpoint_path.parent
        
        # Run a single episode with recording
        results = run_evaluation(
            checkpoint_path,
            n_episodes=1,
            render=False,
            record=True
        )
        
        if results['recordings']:
            plot_network_dynamics(results['recordings'][0], output_path)
            plot_weight_space_3d(results['recordings'][0], output_path)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

