"""
network_models.py — Pydantic schemas for network topology endpoints.
====================================================================
"""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TopologyType(str, Enum):
    MESH = "mesh"
    GRID = "grid"
    RANDOM = "random"


class CongestionLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class TopologyCreateRequest(BaseModel):
    """POST /network — create or reset network topology."""
    topology_type: TopologyType = TopologyType.RANDOM
    num_nodes: int = Field(default=10, ge=3, le=50, description="Number of nodes")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")

    model_config = {"json_schema_extra": {
        "examples": [{"topology_type": "random", "num_nodes": 10, "seed": 42}]
    }}


class EdgeUpdateRequest(BaseModel):
    """PUT /network/edge — update an edge's weight dynamically."""
    source: int = Field(..., ge=0)
    target: int = Field(..., ge=0)
    delay: Optional[float] = Field(default=None, ge=0)
    bandwidth: Optional[float] = Field(default=None, ge=0)
    congestion: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    loss_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class LinkFailureRequest(BaseModel):
    """POST /network/failure — inject or restore a link failure."""
    source: int = Field(..., ge=0)
    target: int = Field(..., ge=0)
    action: str = Field(default="fail", pattern="^(fail|restore)$")


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class NodeResponse(BaseModel):
    id: int
    label: str
    neighbor_count: int


class EdgeResponse(BaseModel):
    source: int
    target: int
    delay: float
    bandwidth: float
    congestion: float
    loss_rate: float
    congestion_level: CongestionLevel
    effective_cost: float


class TopologyMetadata(BaseModel):
    node_count: int
    edge_count: int
    topology_type: str


class TopologyResponse(BaseModel):
    """Full network topology response."""
    nodes: list[NodeResponse]
    edges: list[EdgeResponse]
    metadata: TopologyMetadata


class EdgeUpdateResponse(BaseModel):
    source: int
    target: int
    updated_fields: dict
    new_effective_cost: float


class LinkFailureResponse(BaseModel):
    source: int
    target: int
    status: str  # "removed" | "restored" | "not_found"
    remaining_edges: int
