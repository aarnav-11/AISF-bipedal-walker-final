"""
Numerical integration methods for differential equations.
"""

import numpy as np
from typing import Callable, Any


def runge_kutta_4(
    dt: float,
    state: np.ndarray,
    inputs: Any,
    derivative_fn: Callable[[np.ndarray, Any], np.ndarray]
) -> np.ndarray:
    """
    4th-order Runge-Kutta integration step.
    
    Args:
        dt: Time step size
        state: Current state vector
        inputs: External inputs to the system
        derivative_fn: Function computing state derivatives: f(state, inputs) -> d_state
        
    Returns:
        New state after integration step
    """
    k1 = derivative_fn(state, inputs)
    k2 = derivative_fn(state + 0.5 * dt * k1, inputs)
    k3 = derivative_fn(state + 0.5 * dt * k2, inputs)
    k4 = derivative_fn(state + dt * k3, inputs)
    
    return state + (k1 + 2 * (k2 + k3) + k4) * dt / 6.0


def euler(
    dt: float,
    state: np.ndarray,
    inputs: Any,
    derivative_fn: Callable[[np.ndarray, Any], np.ndarray]
) -> np.ndarray:
    """
    Simple Euler integration step.
    
    Args:
        dt: Time step size
        state: Current state vector
        inputs: External inputs to the system
        derivative_fn: Function computing state derivatives
        
    Returns:
        New state after integration step
    """
    return state + dt * derivative_fn(state, inputs)

