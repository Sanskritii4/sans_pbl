"""
core — Network simulation, Q-learning agent, and routing engines.
=================================================================
Convenience re-exports so other modules can do:
    from backend.core import NetworkSimulator, QLearningAgent
"""

from .network import NetworkSimulator, TopologyConfig
from .agent import QLearningAgent, HyperParameters, RoutingResult
