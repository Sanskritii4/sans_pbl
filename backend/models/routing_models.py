"""
routing_models.py — Pydantic schemas for routing endpoints.
============================================================
"""

from pydantic import BaseModel, Field
from typing import Optional


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class RouteRequest(BaseModel):
    """Query params for GET /route — also usable as body for POST."""
    source: int = Field(..., ge=0, description="Source node ID")
    destination: int = Field(..., ge=0, description="Destination node ID")


class CompareRequest(BaseModel):
    """POST /route/compare — batch comparison."""
    num_packets: int = Field(default=50, ge=1, le=500)
    seed: Optional[int] = None


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class HopDetail(BaseModel):
    from_node: int
    to_node: int
    edge_cost: float
    q_value: Optional[float] = None


class RouteResponse(BaseModel):
    """Single routing result."""
    algorithm: str             # "q-learning" | "dijkstra"
    source: int
    destination: int
    path: list[int]
    total_cost: float
    hop_count: int
    delivered: bool
    reason: str = ""
    per_hop_details: list[HopDetail] = []


class AlgorithmSummary(BaseModel):
    avg_cost: float
    avg_hops: float
    delivery_rate: float
    min_cost: float
    max_cost: float


class ComparisonDelta(BaseModel):
    cost_diff_percent: float
    hop_diff_percent: float
    delivery_diff: float
    winner_by_delivery: str
    winner_by_cost: str


class CompareResponse(BaseModel):
    """Side-by-side comparison of Q-Learning vs Dijkstra."""
    num_packets: int
    q_learning: AlgorithmSummary
    dijkstra: AlgorithmSummary
    delta: ComparisonDelta
