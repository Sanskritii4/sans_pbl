"""
agent_models.py — Pydantic schemas for RL agent / training endpoints.
=====================================================================
"""

from pydantic import BaseModel, Field
from typing import Optional


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class TrainRequest(BaseModel):
    """POST /train — start training the Q-learning agent."""
    episodes: int = Field(default=5000, ge=100, le=50000, description="Number of training episodes")
    alpha: float = Field(default=0.1, ge=0.001, le=1.0, description="Learning rate")
    gamma: float = Field(default=0.95, ge=0.0, le=1.0, description="Discount factor")
    epsilon: float = Field(default=1.0, ge=0.0, le=1.0, description="Initial exploration rate")
    epsilon_min: float = Field(default=0.01, ge=0.0, le=1.0, description="Minimum epsilon")
    epsilon_decay: float = Field(default=0.995, ge=0.9, le=0.9999, description="Epsilon decay rate")
    fluctuate_every: int = Field(default=50, ge=1, le=500, description="Network fluctuation interval")

    model_config = {"json_schema_extra": {
        "examples": [{
            "episodes": 5000, "alpha": 0.1, "gamma": 0.95,
            "epsilon": 1.0, "epsilon_decay": 0.995
        }]
    }}


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class TrainResponse(BaseModel):
    """Response after training completes."""
    status: str
    episodes_completed: int
    q_table_size: int
    final_epsilon: float
    avg_reward_last_500: float
    delivery_rate_last_500: float
    training_time_seconds: float


class HyperParameterResponse(BaseModel):
    alpha: float
    gamma: float
    epsilon_current: float
    epsilon_min: float
    epsilon_decay: float
    max_steps_per_episode: int
    q_table_size: int
    is_trained: bool


class QTableEntry(BaseModel):
    current_node: int
    destination: int
    action: int
    q_value: float


class QTableResponse(BaseModel):
    size: int
    top_entries: list[QTableEntry]
