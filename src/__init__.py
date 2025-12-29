"""
Dynamic Synapse Neural Network for Bipedal Walking

A bio-inspired neural network controller using:
- FitzHugh-Nagumo oscillators as Central Pattern Generators (CPG)
- Dynamic synapses with oscillating weights
- Reward-modulated learning

Based on: "A Bio-inspired Reinforcement Learning Rule to Optimise Dynamical Neural Networks for Robot Control"
https://github.com/InsectRobotics/DynamicSynapseSimplifiedPublic
"""

from .oscillator import FitzHughNagumoOscillator
from .dynamic_synapse import DynamicSynapseLayer
from .network import BipedalController
from .integrators import runge_kutta_4

__all__ = [
    "FitzHughNagumoOscillator",
    "DynamicSynapseLayer", 
    "BipedalController",
    "runge_kutta_4",
]

