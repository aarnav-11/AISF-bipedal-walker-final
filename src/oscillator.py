"""
FitzHugh-Nagumo Neural Oscillator - Central Pattern Generator (CPG)

The FitzHugh-Nagumo model is a simplified version of the Hodgkin-Huxley model
that captures the essential dynamics of neural excitation. It consists of two
coupled differential equations representing membrane voltage (V) and a recovery
variable (W).

This oscillator serves as the Central Pattern Generator (CPG) for producing
rhythmic locomotion patterns in the bipedal walker.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from .integrators import runge_kutta_4


@dataclass
class FitzHughNagumoOscillator:
    """
    FitzHugh-Nagumo oscillator for generating rhythmic patterns.
    
    The dynamics are governed by:
        dV/dt = (V - V³ - W + I) * scale
        dW/dt = a * (b*V - c*W) * scale
    
    Where:
        V: Membrane potential (fast variable)
        W: Recovery variable (slow variable)
        I: External input current
        a, b, c: Model parameters controlling oscillation characteristics
        scale: Time scaling factor
        
    Attributes:
        n_oscillators: Number of independent oscillators
        a: Time scale of recovery variable
        b: Sensitivity of recovery to voltage
        c: Recovery self-decay rate
        scale: Overall time scaling factor
        V: Current voltage states
        W: Current recovery states
        t: Current time
    """
    n_oscillators: int
    a: float = 0.08
    b: float = 2.0
    c: float = 0.8
    scale: float = 0.02
    
    # State variables (initialized in __post_init__)
    V: np.ndarray = field(init=False)
    W: np.ndarray = field(init=False)
    _V_next: np.ndarray = field(init=False, repr=False)
    _W_next: np.ndarray = field(init=False, repr=False)
    t: float = field(default=0.0, init=False)
    
    # Recording
    _history: dict = field(default_factory=dict, init=False, repr=False)
    _recording: bool = field(default=False, init=False, repr=False)
    
    def __post_init__(self):
        """Initialize state arrays."""
        self.V = np.zeros(self.n_oscillators)
        self.W = np.zeros(self.n_oscillators)
        self._V_next = np.zeros(self.n_oscillators)
        self._W_next = np.zeros(self.n_oscillators)
    
    def _derivative(self, state: np.ndarray, I: np.ndarray) -> np.ndarray:
        """
        Compute the derivatives of the FitzHugh-Nagumo system.
        
        Args:
            state: Array [V, W] of current state
            I: External input current
            
        Returns:
            Array [dV/dt, dW/dt] of derivatives
        """
        V, W = state[0], state[1]
        
        # FitzHugh-Nagumo equations
        dV = (V - np.power(V, 3) - W + I) * self.scale
        dW = self.a * (self.b * V - self.c * W) * self.scale
        
        return np.array([dV, dW])
    
    def step(self, dt: float, I: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Advance the oscillator by one time step using RK4 integration.
        
        Args:
            dt: Time step size
            I: External input current for each oscillator
            
        Returns:
            Tuple of (V, W) - current voltage and recovery states
        """
        self.t += dt
        
        # Stack current state
        state = np.array([self.V, self.W])
        
        # Integrate using RK4
        new_state = runge_kutta_4(dt, state, I, self._derivative)
        
        self._V_next = new_state[0]
        self._W_next = new_state[1]
        
        # Clip to prevent numerical instability
        self._V_next = np.clip(self._V_next, -2.0, 2.0)
        self._W_next = np.clip(self._W_next, -2.0, 2.0)
        
        return self._V_next, self._W_next
    
    def update(self):
        """Commit the computed next state as current state."""
        self.V = self._V_next.copy()
        self.W = self._W_next.copy()
        
        if self._recording:
            self._history['t'].append(self.t)
            self._history['V'].append(self.V.copy())
            self._history['W'].append(self.W.copy())
    
    def get_output(self) -> np.ndarray:
        """
        Get the oscillator output suitable for downstream processing.
        
        Returns:
            Flattened array of [V0, V1, ..., W0, W1, ...] for all oscillators
        """
        return np.concatenate([self.V, self.W])
    
    def reset(self, V: Optional[np.ndarray] = None, W: Optional[np.ndarray] = None):
        """
        Reset oscillator state.
        
        Args:
            V: Initial voltage states (default: zeros)
            W: Initial recovery states (default: zeros)
        """
        self.V = V if V is not None else np.zeros(self.n_oscillators)
        self.W = W if W is not None else np.zeros(self.n_oscillators)
        self._V_next = self.V.copy()
        self._W_next = self.W.copy()
        self.t = 0.0
        
        if self._recording:
            self._history = {'t': [], 'V': [], 'W': []}
    
    def start_recording(self):
        """Enable state recording for visualization."""
        self._recording = True
        self._history = {'t': [], 'V': [], 'W': []}
    
    def stop_recording(self):
        """Disable state recording."""
        self._recording = False
    
    def get_history(self) -> dict:
        """Get recorded history as numpy arrays."""
        return {
            't': np.array(self._history['t']),
            'V': np.array(self._history['V']),
            'W': np.array(self._history['W']),
        }
    
    def compute_nullclines(self, I: float = 0.0, v_range: Tuple[float, float] = (-1.5, 1.5)) -> dict:
        """
        Compute nullclines for phase portrait visualization.
        
        Args:
            I: Input current value
            v_range: Range of V values to compute nullclines over
            
        Returns:
            Dictionary with 'v_null' and 'w_null' curves
        """
        V = np.linspace(v_range[0], v_range[1], 100)
        
        # V-nullcline: dV/dt = 0 => W = V - V³ + I
        v_nullcline = V - np.power(V, 3) + I
        
        # W-nullcline: dW/dt = 0 => W = (b/c) * V
        w_nullcline = (self.b / self.c) * V
        
        return {
            'V': V,
            'v_nullcline': v_nullcline,
            'w_nullcline': w_nullcline,
        }


class CPGNetwork:
    """
    Network of coupled FitzHugh-Nagumo oscillators for coordinated locomotion.
    
    This implements a simple Central Pattern Generator network where oscillators
    can be coupled to produce phase-locked rhythmic patterns suitable for
    bipedal locomotion (e.g., anti-phase oscillation for alternating legs).
    """
    
    def __init__(
        self,
        n_oscillators: int = 2,
        coupling_strength: float = 0.1,
        **oscillator_kwargs
    ):
        """
        Initialize CPG network.
        
        Args:
            n_oscillators: Number of oscillators in the network
            coupling_strength: Strength of inter-oscillator coupling
            **oscillator_kwargs: Additional arguments passed to oscillator
        """
        self.n_oscillators = n_oscillators
        self.coupling_strength = coupling_strength
        self.oscillator = FitzHughNagumoOscillator(
            n_oscillators=n_oscillators,
            **oscillator_kwargs
        )
        
        # Default anti-phase coupling for bipedal gait
        self.coupling_matrix = self._create_alternating_coupling()
    
    def _create_alternating_coupling(self) -> np.ndarray:
        """Create coupling matrix for alternating (anti-phase) gait."""
        # Simple inhibitory coupling between alternating oscillators
        coupling = np.zeros((self.n_oscillators, self.n_oscillators))
        for i in range(self.n_oscillators):
            for j in range(self.n_oscillators):
                if i != j:
                    coupling[i, j] = -self.coupling_strength
        return coupling
    
    def step(self, dt: float, external_input: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Step the CPG network forward.
        
        Args:
            dt: Time step
            external_input: External input to each oscillator
            
        Returns:
            Tuple of (V, W) states
        """
        # Add inter-oscillator coupling to external input
        coupled_input = external_input + self.coupling_matrix @ self.oscillator.V
        
        return self.oscillator.step(dt, coupled_input)
    
    def update(self):
        """Update oscillator states."""
        self.oscillator.update()
    
    def get_output(self) -> np.ndarray:
        """Get combined output from all oscillators."""
        return self.oscillator.get_output()
    
    def reset(self):
        """Reset the CPG network."""
        self.oscillator.reset()


if __name__ == "__main__":
    # Simple test of the oscillator
    import matplotlib.pyplot as plt
    
    # Create oscillator
    osc = FitzHughNagumoOscillator(n_oscillators=2, scale=0.02)
    osc.start_recording()
    
    # Simulate with varying input
    T = 10000  # ms
    dt = 33  # ms
    n_steps = int(T / dt)
    
    for step in range(n_steps):
        t = step * dt
        # Varying input over time
        I = np.array([0.5 * np.sin(2 * np.pi * t / 5000), 
                      0.5 * np.sin(2 * np.pi * t / 5000 + np.pi)])
        osc.step(dt, I)
        osc.update()
    
    # Plot results
    history = osc.get_history()
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    
    # Time series
    axes[0].plot(history['t'], history['V'][:, 0], label='V₁')
    axes[0].plot(history['t'], history['V'][:, 1], label='V₂')
    axes[0].set_xlabel('Time (ms)')
    axes[0].set_ylabel('Voltage')
    axes[0].legend()
    axes[0].set_title('FitzHugh-Nagumo Oscillator Output')
    
    # Phase portrait
    axes[1].plot(history['V'][:, 0], history['W'][:, 0], label='Oscillator 1')
    axes[1].plot(history['V'][:, 1], history['W'][:, 1], label='Oscillator 2')
    axes[1].set_xlabel('V')
    axes[1].set_ylabel('W')
    axes[1].legend()
    axes[1].set_title('Phase Portrait')
    
    plt.tight_layout()
    plt.savefig('oscillator_test.png', dpi=150)
    plt.show()

