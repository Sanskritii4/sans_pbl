"""
metrics_models.py — Pydantic schemas for metrics / analytics endpoints.
=======================================================================
"""

from pydantic import BaseModel
from typing import Optional


class EpisodeMetricPoint(BaseModel):
    episode: int
    reward: float
    steps: int
    delivered: bool
    epsilon: float


class MetricsSummary(BaseModel):
    """GET /metrics — aggregate training + routing metrics."""
    is_trained: bool
    total_episodes: int
    q_table_size: int

    # Training convergence
    avg_reward_first_500: Optional[float] = None
    avg_reward_last_500: Optional[float] = None
    reward_improvement_percent: Optional[float] = None
    delivery_rate_first_500: Optional[float] = None
    delivery_rate_last_500: Optional[float] = None

    # Recent reward history (for charting)
    reward_history: list[EpisodeMetricPoint] = []

    # Network state
    network_nodes: int = 0
    network_edges: int = 0
    topology_type: str = ""
