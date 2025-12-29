"""
Dynamic Synapse Layer with Oscillating Weights

This implements the core learning mechanism from the paper:
"A Bio-inspired Reinforcement Learning Rule to Optimise Dynamical Neural Networks for Robot Control"

The key insight is that synaptic weights oscillate around a center value,
and learning occurs by shifting this center based on reward signals.
When a positive reward is received, the current weight deviation from center
is used to update the center, effectively "capturing" beneficial weight configurations.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import copy


@dataclass
class DynamicSynapseLayer:
    """
    A layer of synapses with oscillating weights and reward-modulated learning.
    
    The weight at time t is:
        W(t) = W_center + Amplitude * sin(2π * t_in_period / Period)
    
    Learning rule:
        dW_center/dt = (W - W_center) * reward * learning_rate
        
    This means:
        - Positive reward: Center moves toward current weight
        - Negative reward: Center moves away from current weight (repulsive)
        - The oscillation amplitude decays with accumulated positive reward
    
    Attributes:
        shape: (n_outputs, n_inputs) - shape of the weight matrix
        period: Base oscillation period (randomized per synapse)
        period_variance: Variance in period (for desynchronization)
        amplitude: Initial oscillation amplitude
        learning_rate: Rate of center update
        amplitude_decay: Rate at which amplitude decays with reward
    """
    shape: Tuple[int, int]
    period: float = 20000.0
    period_variance: float = 0.1
    amplitude: float = 0.2
    learning_rate: float = 0.000012
    amplitude_decay: float = 0.0000003
    initial_weight_scale: float = 0.4
    
    # Internal state
    weights: np.ndarray = field(init=False)
    weights_center: np.ndarray = field(init=False)
    weights_last: np.ndarray = field(init=False)
    _periods: np.ndarray = field(init=False, repr=False)
    _period_centers: np.ndarray = field(init=False, repr=False)
    _t_in_period: np.ndarray = field(init=False, repr=False)
    _amplitudes: np.ndarray = field(init=False, repr=False)
    t: float = field(default=0.0, init=False)
    
    # Recording
    _history: dict = field(default_factory=dict, init=False, repr=False)
    _recording: bool = field(default=False, init=False, repr=False)
    
    def __post_init__(self):
        """Initialize weight arrays."""
        self._initialize_weights()
    
    def _initialize_weights(self, weights_center: Optional[np.ndarray] = None):
        """
        Initialize or reset all weight-related arrays.
        
        Args:
            weights_center: Optional preset center weights
        """
        n_out, n_in = self.shape
        
        # Initialize periods with randomization for desynchronization
        self._period_centers = np.ones(self.shape) * self.period
        self._periods = self._period_centers.copy()
        
        # Random phase initialization
        self._t_in_period = np.random.rand(n_out, n_in) * self._periods
        
        # Amplitude per synapse
        self._amplitudes = np.ones(self.shape) * self.amplitude
        
        # Weight centers (learned parameters)
        if weights_center is not None:
            self.weights_center = weights_center.copy()
        else:
            self.weights_center = (np.random.rand(n_out, n_in) - 0.5) * self.initial_weight_scale
        
        # Current weights (oscillating around center)
        self._update_weights()
        self.weights_last = self.weights.copy()
        
        self.t = 0.0
    
    def _update_weights(self):
        """Update weights based on oscillation."""
        phase = 2 * np.pi * self._t_in_period / self._periods
        self.weights = self.weights_center + self._amplitudes * np.sin(phase)
    
    def step(self, dt: float, reward: float) -> np.ndarray:
        """
        Step the dynamic synapse forward in time.
        
        Args:
            dt: Time step
            reward: Reward signal (positive reinforces, negative repulses)
            
        Returns:
            Current weight matrix
        """
        self.t += dt
        self.weights_last = self.weights.copy()
        
        # Advance time within period
        self._t_in_period += dt
        
        # Update weights based on oscillation
        self._update_weights()
        
        # Learning: shift center toward current weight if rewarded
        weight_deviation = self.weights - self.weights_center
        self.weights_center += weight_deviation * reward * self.learning_rate * dt
        
        # Amplitude decay with positive reward (exploitation vs exploration)
        if reward > 0:
            self._amplitudes *= np.exp(-self.amplitude_decay * reward * dt)
        
        # Period reset at zero crossing (prevents phase drift)
        zero_crossing = np.logical_and(
            self.weights_last < self.weights_center,
            self.weights >= self.weights_center
        )
        
        # Reset period with slight randomization at zero crossings
        if np.any(zero_crossing):
            self._t_in_period[zero_crossing] = self._t_in_period[zero_crossing] % self._periods[zero_crossing]
            # Add period variance for desynchronization
            self._periods[zero_crossing] = np.random.normal(
                loc=self._period_centers[zero_crossing],
                scale=self._period_centers[zero_crossing] * self.period_variance
            )
        
        if self._recording:
            self._record_state(reward)
        
        return self.weights
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the layer.
        
        Args:
            x: Input vector of shape (n_inputs,)
            
        Returns:
            Output vector of shape (n_outputs,)
        """
        return self.weights @ x
    
    def _record_state(self, reward: float):
        """Record current state for visualization."""
        self._history['t'].append(self.t)
        self._history['weights'].append(self.weights.copy())
        self._history['weights_center'].append(self.weights_center.copy())
        self._history['amplitudes'].append(self._amplitudes.copy())
        self._history['reward'].append(reward)
    
    def start_recording(self):
        """Enable state recording."""
        self._recording = True
        self._history = {
            't': [],
            'weights': [],
            'weights_center': [],
            'amplitudes': [],
            'reward': [],
        }
    
    def stop_recording(self):
        """Disable state recording."""
        self._recording = False
    
    def get_history(self) -> dict:
        """Get recorded history as numpy arrays."""
        return {k: np.array(v) for k, v in self._history.items()}
    
    def reset(self, weights_center: Optional[np.ndarray] = None):
        """
        Reset the layer state.
        
        Args:
            weights_center: Optional preset center weights
        """
        self._initialize_weights(weights_center)
        if self._recording:
            self._history = {
                't': [],
                'weights': [],
                'weights_center': [],
                'amplitudes': [],
                'reward': [],
            }
    
    def get_state(self) -> dict:
        """Get current state for saving."""
        return {
            'weights_center': self.weights_center.copy(),
            'amplitudes': self._amplitudes.copy(),
            'periods': self._periods.copy(),
            't_in_period': self._t_in_period.copy(),
            't': self.t,
        }
    
    def set_state(self, state: dict):
        """Restore state from saved data."""
        self.weights_center = state['weights_center'].copy()
        self._amplitudes = state['amplitudes'].copy()
        self._periods = state['periods'].copy()
        self._t_in_period = state['t_in_period'].copy()
        self.t = state['t']
        self._update_weights()
        self.weights_last = self.weights.copy()


class GainLayer:
    """
    Simple gain/scaling layer for signal processing.
    
    Used in the preprocessing stage of the network.
    """
    
    def __init__(self, gain: float = 1.0):
        """
        Initialize gain layer.
        
        Args:
            gain: Multiplicative gain factor
        """
        self.gain = gain
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply gain to input."""
        return x * self.gain


class AdaptiveNeuronLayer:
    """
    Layer of neurons with adaptive sensitivity.
    
    The sensitivity adapts to maintain outputs in a target range,
    preventing saturation and ensuring gradient flow.
    """
    
    def __init__(
        self,
        n_neurons: int,
        initial_sensitivity: float = 0.5,
        adaptation_rate: float = 0.000001,
        target_output: float = 0.3,
        use_nonlinear_adaptation: bool = True
    ):
        """
        Initialize adaptive neuron layer.
        
        Args:
            n_neurons: Number of neurons
            initial_sensitivity: Initial sensitivity value
            adaptation_rate: Rate of sensitivity adaptation
            target_output: Target absolute output level
            use_nonlinear_adaptation: Whether to use nonlinear adaptation
        """
        self.n_neurons = n_neurons
        self.sensitivity = np.ones(n_neurons) * initial_sensitivity
        self.adaptation_rate = adaptation_rate
        self.target_output = target_output
        self.use_nonlinear_adaptation = use_nonlinear_adaptation
        
        self._last_output = np.zeros(n_neurons)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass with ReLU of tanh.
        
        Args:
            x: Input vector
            
        Returns:
            Activated output
        """
        # Scale by sensitivity, apply tanh, then ReLU
        scaled = np.tanh(x * self.sensitivity)
        self._last_output = np.maximum(scaled, 0)
        return self._last_output
    
    def adapt(self, dt: float):
        """
        Adapt sensitivity based on recent outputs.
        
        Args:
            dt: Time step
        """
        error = self.target_output - np.abs(self._last_output)
        
        if self.use_nonlinear_adaptation:
            # Nonlinear adaptation: faster when sensitivity is far from 1
            log_sens = np.log10(np.clip(self.sensitivity, 1e-3, 1e3))
            adaptation_factor = (2 - log_sens) * (log_sens + 2) / 4
            self.sensitivity += error * self.adaptation_rate * dt * adaptation_factor
        else:
            self.sensitivity += error * self.adaptation_rate * dt
        
        # Clip to reasonable range (cap at 5.0 to prevent runaway)
        self.sensitivity = np.clip(self.sensitivity, 1e-3, 5.0)
    
    def reset(self, initial_sensitivity: Optional[float] = None):
        """Reset sensitivity to initial value."""
        if initial_sensitivity is not None:
            self.sensitivity = np.ones(self.n_neurons) * initial_sensitivity
        else:
            self.sensitivity = np.ones(self.n_neurons) * 0.5
        self._last_output = np.zeros(self.n_neurons)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Test dynamic synapse layer
    layer = DynamicSynapseLayer(
        shape=(4, 10),
        period=2000,
        amplitude=0.3,
        learning_rate=0.001,
    )
    layer.start_recording()
    
    # Simulate with varying reward
    dt = 33
    T = 20000
    n_steps = int(T / dt)
    
    for step in range(n_steps):
        t = step * dt
        # Reward signal (periodic bursts)
        reward = 1.0 if (t % 5000) < 1000 else 0.0
        layer.step(dt, reward)
    
    # Plot results
    history = layer.get_history()
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    
    # Weight evolution
    axes[0].plot(history['t'], history['weights'][:, 0, 0], label='Weight')
    axes[0].plot(history['t'], history['weights_center'][:, 0, 0], label='Center', linestyle='--')
    axes[0].set_xlabel('Time (ms)')
    axes[0].set_ylabel('Weight')
    axes[0].legend()
    axes[0].set_title('Dynamic Synapse Weight Evolution')
    
    # Reward
    axes[1].plot(history['t'], history['reward'])
    axes[1].set_xlabel('Time (ms)')
    axes[1].set_ylabel('Reward')
    axes[1].set_title('Reward Signal')
    
    # Amplitude decay
    axes[2].plot(history['t'], history['amplitudes'][:, 0, 0])
    axes[2].set_xlabel('Time (ms)')
    axes[2].set_ylabel('Amplitude')
    axes[2].set_title('Oscillation Amplitude')
    
    plt.tight_layout()
    plt.savefig('dynamic_synapse_test.png', dpi=150)
    plt.show()

