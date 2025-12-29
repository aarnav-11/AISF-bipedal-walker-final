# Dynamic Synapse Neural Network for Bipedal Walking

A bio-inspired neural network controller for the BipedalWalker-v3 environment using **oscillating synaptic weights** and **FitzHugh-Nagumo oscillators** as Central Pattern Generators (CPG).

## Demo

Trained walker achieving 300+ reward:

<video src="docs/videos/walker_pass.mp4" controls width="600"></video>

<video src="docs/videos/walker_pass_2.mp4" controls width="600"></video>

> **Note:** If videos don't play in GitHub, download them from [docs/videos/](docs/videos/) or clone the repo.

*Videos show the walker successfully traversing the terrain after ~50,000 episodes of training.*

## Overview

This implementation is based on the research paper:
> **"A Bio-inspired Reinforcement Learning Rule to Optimise Dynamical Neural Networks for Robot Control"** (IROS 2018)
> 
> Original repository: [InsectRobotics/DynamicSynapseSimplifiedPublic](https://github.com/InsectRobotics/DynamicSynapseSimplifiedPublic)

### Key Concepts

**Dynamic Synapses**: Unlike traditional neural networks with static weights, this approach uses weights that **oscillate** around a learned center value:

```
W(t) = W_center + Amplitude × sin(2π × t / Period)
```

**Learning Rule**: When positive reward is received, the center shifts toward the current weight value:

```
dW_center/dt = (W - W_center) × reward × learning_rate
```

This creates a form of **reward-modulated Hebbian learning** where:
- Positive reward reinforces current weight configurations
- Negative reward (with repulsive learning) pushes weights away from current values
- Oscillation amplitude decays over time as learning converges

**Central Pattern Generator (CPG)**: FitzHugh-Nagumo neural oscillators generate rhythmic patterns for coordinated leg movement, inspired by biological locomotion systems.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           PREPROCESSING                                  │
├─────────────────────────────────────────────────────────────────────────┤
│  States (14) ──→ ReLU ──→ 14 positive                                   │
│  States (14) × -1 ──→ -ReLU ──→ 14 negative                            │
│  Lidar (10) ──→ 10                                                      │
│  Constant (1)                                                           │
│  Total: 39 dims                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                            LAYER 1                                       │
├─────────────────────────────────────────────────────────────────────────┤
│  Input: 39 (preprocessed) + 4 (action feedback) = 43                   │
│  Dynamic Synapse Layer: 43 → 8                                          │
│  Activation: tanh                                                        │
├─────────────────────────────────────────────────────────────────────────┤
│                            LAYER 2                                       │
├─────────────────────────────────────────────────────────────────────────┤
│  8 neurons with adaptive sensitivity                                    │
│  Activation: ReLU(tanh(x × sensitivity))                               │
├─────────────────────────────────────────────────────────────────────────┤
│                         LAYER 3 (CPG)                                    │
├─────────────────────────────────────────────────────────────────────────┤
│  Dynamic Synapse: 9 → 2 (CPG inputs)                                   │
│  2 FitzHugh-Nagumo Oscillators                                          │
│  Output: 4 (V₀, V₁, W₀, W₁)                                            │
├─────────────────────────────────────────────────────────────────────────┤
│                            LAYER 4                                       │
├─────────────────────────────────────────────────────────────────────────┤
│  Input: 4 (CPG) + 14 (states) + 1 (bias) = 19                          │
│  Dynamic Synapse Layer: 19 → 4                                          │
│  Activation: tanh → 4 actions                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Clone the repository
cd bipedal2

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training

```bash
# Basic training
python train.py

# With custom options
python train.py --episodes 50000 --seed 42 --repulsive True --adaptation nonlinear

# With rendering (slower but visualized)
python train.py --render --episodes 1000
```

**Training Options:**
- `--episodes`: Number of training episodes (default: 10000)
- `--render`: Enable visualization during training
- `--repulsive`: Enable repulsive learning for negative rewards (default: True)
- `--adaptation`: Neuron sensitivity adaptation type: none, linear, nonlinear
- `--seed`: Random seed for reproducibility
- `--save-every`: Checkpoint save frequency

### Evaluation & Visualization

```bash
# Plot training progress
python visualize.py plot results/run_xxx/checkpoint_best.pkl

# Run evaluation episodes with rendering
python visualize.py eval results/run_xxx/checkpoint_best.pkl --episodes 5

# Visualize network dynamics
python visualize.py dynamics results/run_xxx/checkpoint_best.pkl
```

## Project Structure

```
bipedal2/
├── src/
│   ├── __init__.py           # Package exports
│   ├── integrators.py        # Numerical integration (RK4)
│   ├── oscillator.py         # FitzHugh-Nagumo CPG
│   ├── dynamic_synapse.py    # Oscillating weight layers
│   └── network.py            # Full controller architecture
├── train.py                  # Training script
├── visualize.py              # Visualization utilities
├── requirements.txt          # Python dependencies
└── README.md
```

## How It Works

### 1. Weight Oscillation

Each synapse has a weight that oscillates sinusoidally:
- **Period**: ~20 seconds (randomized per synapse for desynchronization)
- **Amplitude**: Starts at 0.2, decays with positive reward (exploitation)
- **Center**: Learned value that shifts based on reward

### 2. Reward-Modulated Learning

The learning rule is elegant:
```python
weight_deviation = current_weight - weight_center
weight_center += weight_deviation * reward * learning_rate * dt
```

This means:
- If the current (oscillating) weight leads to good reward, the center moves toward it
- The oscillation acts as **exploration** of weight space
- As amplitude decays, the network **exploits** learned weights

### 3. Central Pattern Generator

The FitzHugh-Nagumo oscillators naturally produce alternating patterns ideal for bipedal gait:
- Two coupled oscillators generate anti-phase rhythms
- External input modulates oscillation frequency and amplitude
- Outputs drive the four joint actuators

### 4. Adaptive Neurons

The sensitivity of hidden neurons adapts to maintain outputs in a useful range:
- Prevents saturation
- Enables faster learning in new situations
- Nonlinear adaptation is more stable

## Results

The network typically:
- Shows improvement within the first 1000 episodes
- Achieves stable walking after ~5000-10000 episodes
- The "solved" threshold (mean reward > 300 over 100 episodes) may take 20000+ episodes

## References

1. [A Bio-inspired Reinforcement Learning Rule to Optimise Dynamical Neural Networks for Robot Control](https://www.research.ed.ac.uk/portal/en/publications/a-bioinspired-reinforcement-learning-rule-to-optimise-dynamical-neural-networks-for-robot-control.html) - IROS 2018

2. [A model of operant learning based on chaotically varying synaptic strength](https://doi.org/10.1016/j.neunet.2018.08.006) - Neural Networks 2018

3. [FitzHugh-Nagumo Model](https://en.wikipedia.org/wiki/FitzHugh%E2%80%93Nagumo_model) - Wikipedia

4. [Video Demo](https://youtu.be/B7mLVY1NKgI) - Original paper demonstration

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- Original implementation by [InsectRobotics](https://github.com/InsectRobotics)
- BipedalWalker environment by [OpenAI Gym](https://gymnasium.farama.org/)

