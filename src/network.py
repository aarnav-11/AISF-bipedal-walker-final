"""
Bipedal Walker Controller Neural Network

This implements the full network architecture from the diagram:

Preprocessing → Layer 1 → Layer 2 → Layer 3 (CPG) → Layer 4 → Output

Key components:
- Preprocessing: Separate positive/negative state channels + lidar
- Layer 1: Merge all inputs with dynamic synapses (43 → 8)
- Layer 2: Neuron sensitivity adaptation (8 neurons)
- Layer 3: FitzHugh-Nagumo oscillators (CPG) - 2 oscillators → 4 outputs
- Layer 4: Final merge layer (18 → 4)
- Output: 4 action values (joint torques)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any
from .oscillator import FitzHughNagumoOscillator
from .dynamic_synapse import DynamicSynapseLayer, AdaptiveNeuronLayer


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation function."""
    return np.maximum(x, 0)


def tanh(x: np.ndarray) -> np.ndarray:
    """Tanh activation function."""
    return np.tanh(x)


class ObservationPreprocessor:
    """
    Preprocesses BipedalWalker observations.
    
    The BipedalWalker-v3 observation space has 24 dimensions:
    - 0-13: Robot state (hull angle, velocities, joint angles, leg contacts, etc.)
    - 14-23: Lidar rangefinder readings (10 values)
    
    This preprocessor:
    1. Normalizes observations using hand-tuned factors
    2. Splits states into positive (ReLU) and negative (-ReLU) channels
    3. Appends lidar readings and a bias term
    
    Output: 14 (positive) + 14 (negative) + 10 (lidar) + 1 (bias) = 39 dims
    """
    
    # Normalization factors for each observation dimension
    NORM_FACTORS = np.array([
        1 / (2 * np.pi),  # 0: hull angle
        5.0,              # 1: hull angular velocity
        2.0,              # 2: hull x velocity
        2.0,              # 3: hull y velocity
        1.0,              # 4: hip 1 angle
        0.5,              # 5: hip 1 speed
        1.0,              # 6: knee 1 angle
        0.3,              # 7: knee 1 speed
        1.0,              # 8: leg 1 ground contact
        1.0,              # 9: hip 2 angle
        0.5,              # 10: hip 2 speed
        1.0,              # 11: knee 2 angle
        0.25,             # 12: knee 2 speed
        1.0,              # 13: leg 2 ground contact
        2.0,              # 14-23: lidar (repeated for remaining)
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
    ])
    
    def __init__(self, gain: float = -1.0):
        """
        Initialize preprocessor.
        
        Args:
            gain: Gain applied to negative channel
        """
        self.gain = gain
    
    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """
        Preprocess observation.
        
        Args:
            obs: Raw observation from BipedalWalker-v3 (24 dims)
            
        Returns:
            Preprocessed observation (39 dims)
        """
        # Normalize
        normalized = obs * self.NORM_FACTORS[:len(obs)]
        
        # Split state part (first 14 dims) into positive and negative
        states = normalized[:14]
        positive = relu(states)
        negative = self.gain * relu(-states)  # Negative channel with gain
        
        # Lidar readings (dims 14-23)
        lidar = normalized[14:] if len(obs) > 14 else np.array([])
        
        # Bias term
        bias = np.array([1.0])
        
        return np.concatenate([positive, negative, lidar, bias])
    
    @property
    def output_dim(self) -> int:
        """Output dimension after preprocessing."""
        return 14 + 14 + 10 + 1  # 39


@dataclass
class BipedalController:
    """
    Complete neural network controller for BipedalWalker.
    
    Architecture (matching the diagram):
    
    Preprocessing:
        - States (14) → ReLU → 14
        - States (14) × -1 → -ReLU → 14  
        - Lidar (10) → 10
        - Constant (1)
        Total: 39 inputs
    
    Layer 1 (Merge + Dynamic Synapse):
        - Input: 39 + 4 (action feedback) = 43
        - Dynamic synapse layer: 43 → 8
        - Activation: tanh
        
    Layer 2 (Neuron Sensitivity):
        - 8 neurons with adaptive sensitivity
        - Activation: ReLU(tanh(x * sensitivity))
        
    Layer 3 (CPG - FitzHugh-Nagumo):
        - Dynamic synapse: 8+1 → 2 (CPG inputs)
        - 2 FN oscillators → 4 outputs (V0, V1, W0, W1)
        
    Layer 4 (Output Merge):
        - Input: 4 (CPG) + 14 (states) + 1 (bias) = 19
        - Dynamic synapse: 19 → 4
        - Activation: tanh → Action
    """
    
    # Network hyperparameters
    dt: float = 33.0  # Time step in ms
    period: float = 20000.0  # Oscillation period
    learning_rate: float = 0.00012  # 10x faster learning
    amplitude: float = 0.2
    amplitude_decay: float = 0.00000003  # 10x slower decay (explore longer)
    sensitivity_adaptation_rate: float = 0.000001
    repulsive_learning: bool = True
    use_nonlinear_adaptation: bool = True
    
    # Network components (initialized in __post_init__)
    preprocessor: ObservationPreprocessor = field(init=False)
    layer1: DynamicSynapseLayer = field(init=False)
    layer2_neurons: AdaptiveNeuronLayer = field(init=False)
    layer2_synapse: DynamicSynapseLayer = field(init=False)
    cpg: FitzHughNagumoOscillator = field(init=False)
    layer4: DynamicSynapseLayer = field(init=False)
    
    # State
    t: float = field(default=0.0, init=False)
    last_action: np.ndarray = field(init=False)
    
    # Dimensions
    n_actions: int = 4
    n_states: int = 14
    n_lidar: int = 10
    
    def __post_init__(self):
        """Initialize all network components."""
        self._build_network()
    
    def _build_network(self):
        """Construct the neural network architecture."""
        # Preprocessor
        self.preprocessor = ObservationPreprocessor(gain=-1.0)
        
        # Layer 1: Main dynamic synapse layer
        # Input: preprocessed obs (39) + action feedback (4) = 43
        # Output: 8 (for neuron sensitivity layer)
        layer1_in = 39 + self.n_actions
        layer1_out = 8
        self.layer1 = DynamicSynapseLayer(
            shape=(layer1_out, layer1_in),
            period=self.period,
            amplitude=self.amplitude,
            learning_rate=self.learning_rate,
            amplitude_decay=self.amplitude_decay,
        )
        
        # Layer 2: Adaptive neurons
        self.layer2_neurons = AdaptiveNeuronLayer(
            n_neurons=layer1_out,
            initial_sensitivity=0.5,
            adaptation_rate=self.sensitivity_adaptation_rate,
            use_nonlinear_adaptation=self.use_nonlinear_adaptation,
        )
        
        # Layer 2 synapse: 8+1 → 2 (CPG inputs)
        self.layer2_synapse = DynamicSynapseLayer(
            shape=(2, layer1_out + 1),
            period=self.period,
            amplitude=self.amplitude,
            learning_rate=self.learning_rate,
            amplitude_decay=self.amplitude_decay,
        )
        
        # Layer 3: CPG (FitzHugh-Nagumo oscillators)
        self.cpg = FitzHughNagumoOscillator(
            n_oscillators=2,
            scale=0.02,
        )
        
        # Layer 4: Output layer
        # Input: CPG (4) + states (14) + bias (1) = 19
        # Output: 4 actions
        # Initialize with structured weights for coordinated leg movement
        layer4_center = self._create_layer4_initial_weights()
        self.layer4 = DynamicSynapseLayer(
            shape=(self.n_actions, 4 + self.n_states + 1),
            period=self.period,
            amplitude=self.amplitude,
            learning_rate=self.learning_rate,
            amplitude_decay=self.amplitude_decay,
        )
        self.layer4.weights_center = layer4_center
        self.layer4._update_weights()
        
        # Initialize action
        self.last_action = np.zeros(self.n_actions)
        self.t = 0.0
    
    def _create_layer4_initial_weights(self) -> np.ndarray:
        """
        Create initial weights for Layer 4 with structure for coordinated movement.
        
        The weights are initialized to create a reasonable mapping from
        CPG oscillations and state feedback to joint commands.
        """
        # Shape: (4 actions, 19 inputs)
        # Inputs: [V0, V1, W0, W1, 14 states, bias]
        weights = np.zeros((self.n_actions, 4 + self.n_states + 1))
        
        # Action 0 (hip 1): driven by oscillator 0, state feedback from hip 1
        weights[0, 2] = -0.5   # W0 (recovery gives timing)
        weights[0, 8] = -0.2   # Hip 1 angle feedback
        weights[0, 9] = -0.2   # Hip 1 speed feedback
        
        # Action 1 (knee 1): driven by oscillator 0, offset phase
        weights[1, 0] = 0.5    # V0
        weights[1, 10] = -0.2  # Knee 1 angle feedback
        weights[1, 11] = -0.2  # Knee 1 speed feedback
        
        # Action 2 (hip 2): driven by oscillator 1 (anti-phase)
        weights[2, 3] = -0.5   # W1
        weights[2, 13] = -0.2  # Hip 2 angle feedback
        weights[2, 14] = -0.2  # Hip 2 speed feedback
        
        # Action 3 (knee 2): driven by oscillator 1
        weights[3, 1] = 0.5    # V1
        weights[3, 15] = -0.2  # Knee 2 angle feedback
        weights[3, 16] = -0.2  # Knee 2 speed feedback
        
        return weights
    
    def step(self, observation: np.ndarray, reward: float) -> np.ndarray:
        """
        Compute action given observation and update network with reward.
        
        Args:
            observation: Raw observation from BipedalWalker (24 dims)
            reward: Reward from previous action
            
        Returns:
            Action vector (4 dims, each in [-1, 1])
        """
        self.t += self.dt
        
        # Apply repulsive learning (or not)
        if not self.repulsive_learning:
            reward = max(0.0, reward)
        
        # === Preprocessing ===
        preprocessed = self.preprocessor(observation)
        
        # === Layer 1: Merge inputs ===
        layer1_input = np.concatenate([preprocessed, self.last_action])
        
        # Step dynamic synapses with reward
        self.layer1.step(self.dt, reward)
        layer1_out = tanh(self.layer1.forward(layer1_input))
        
        # === Layer 2: Neuron sensitivity ===
        layer2_out = self.layer2_neurons.forward(layer1_out)
        self.layer2_neurons.adapt(self.dt)
        
        # CPG input (add bias)
        self.layer2_synapse.step(self.dt, reward)
        cpg_input = tanh(self.layer2_synapse.forward(np.append(layer2_out, 1.0)))
        
        # === Layer 3: CPG ===
        self.cpg.step(self.dt, cpg_input)
        self.cpg.update()
        cpg_output = self.cpg.get_output()  # [V0, V1, W0, W1]
        
        # === Layer 4: Output ===
        layer4_input = np.concatenate([
            cpg_output,
            observation[:self.n_states],  # Raw state feedback
            np.array([1.0])  # Bias
        ])
        
        self.layer4.step(self.dt, reward)
        action = tanh(self.layer4.forward(layer4_input))
        
        self.last_action = action
        return action
    
    def reset(self):
        """Full reset - rebuilds network from scratch. Use for new training runs."""
        self._build_network()
    
    def soft_reset(self):
        """
        Soft reset for episode boundaries.
        
        Resets CPG state and action buffer but PRESERVES learned weights.
        This should be called at the start of each episode during training.
        """
        self.cpg.reset()
        self.last_action = np.zeros(self.n_actions)
        # Don't reset t - keep global time for oscillation continuity
    
    def get_state(self) -> Dict[str, Any]:
        """Get current network state for saving."""
        return {
            'layer1': self.layer1.get_state(),
            'layer2_synapse': self.layer2_synapse.get_state(),
            'layer4': self.layer4.get_state(),
            'layer2_sensitivity': self.layer2_neurons.sensitivity.copy(),
            't': self.t,
        }
    
    def set_state(self, state: Dict[str, Any]):
        """Restore network state from saved data."""
        self.layer1.set_state(state['layer1'])
        self.layer2_synapse.set_state(state['layer2_synapse'])
        self.layer4.set_state(state['layer4'])
        self.layer2_neurons.sensitivity = state['layer2_sensitivity'].copy()
        self.t = state['t']
    
    def start_recording(self):
        """Enable recording on all layers."""
        self.layer1.start_recording()
        self.layer2_synapse.start_recording()
        self.layer4.start_recording()
        self.cpg.start_recording()
    
    def stop_recording(self):
        """Disable recording on all layers."""
        self.layer1.stop_recording()
        self.layer2_synapse.stop_recording()
        self.layer4.stop_recording()
        self.cpg.stop_recording()
    
    def get_recording(self) -> Dict[str, Any]:
        """Get recorded data from all layers."""
        return {
            'layer1': self.layer1.get_history(),
            'layer2_synapse': self.layer2_synapse.get_history(),
            'layer4': self.layer4.get_history(),
            'cpg': self.cpg.get_history(),
        }


if __name__ == "__main__":
    # Simple test
    controller = BipedalController()
    
    # Simulate with random observations
    for i in range(100):
        obs = np.random.randn(24) * 0.1
        reward = np.random.randn() * 0.1
        action = controller.step(obs, reward)
        
        if i % 20 == 0:
            print(f"Step {i}: action = {action}")
    
    print("\nController test passed!")

