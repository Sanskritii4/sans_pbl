"""
router_agent.py — API endpoints for RL agent training & configuration.
=======================================================================
Prefix: /train  (and /agent)
"""

from fastapi import APIRouter, HTTPException
from backend.models.agent_models import (
    TrainRequest, TrainResponse, HyperParameterResponse, QTableResponse, QTableEntry,
)
from backend.services.routing_service import get_service

router = APIRouter(tags=["Agent / Training"])


@router.post(
    "/train",
    response_model=TrainResponse,
    summary="Train the Q-learning agent",
    response_description="Training summary with convergence metrics",
)
def train_agent(req: TrainRequest):
    """
    Train (or retrain) the Q-learning agent on the current network topology.

    Training is synchronous — the response is returned once training finishes.
    For a 10-node graph with 5000 episodes, this typically takes 2–5 seconds.

    **Example request:**
    ```json
    {
      "episodes": 5000,
      "alpha": 0.1,
      "gamma": 0.95,
      "epsilon": 1.0,
      "epsilon_decay": 0.995
    }
    ```

    **Example response:**
    ```json
    {
      "status": "completed",
      "episodes_completed": 5000,
      "q_table_size": 371,
      "final_epsilon": 0.010000,
      "avg_reward_last_500": 67.3,
      "delivery_rate_last_500": 98.4,
      "training_time_seconds": 3.21
    }
    ```
    """
    svc = get_service()
    result = svc.train_agent(
        episodes=req.episodes,
        alpha=req.alpha,
        gamma=req.gamma,
        epsilon=req.epsilon,
        epsilon_min=req.epsilon_min,
        epsilon_decay=req.epsilon_decay,
        fluctuate_every=req.fluctuate_every,
    )
    return TrainResponse(**result)


@router.get(
    "/agent/hyperparameters",
    response_model=HyperParameterResponse,
    summary="Get current hyperparameter configuration",
)
def get_hyperparameters():
    """Return the agent's current hyperparameters and training status."""
    svc = get_service()
    if svc.agent is None:
        return HyperParameterResponse(
            alpha=0.1, gamma=0.95, epsilon_current=1.0,
            epsilon_min=0.01, epsilon_decay=0.995,
            max_steps_per_episode=50, q_table_size=0, is_trained=False,
        )
    hp = svc.agent.hp
    return HyperParameterResponse(
        alpha=hp.alpha,
        gamma=hp.gamma,
        epsilon_current=round(svc.agent._epsilon, 6),
        epsilon_min=hp.epsilon_min,
        epsilon_decay=hp.epsilon_decay,
        max_steps_per_episode=hp.max_steps_per_episode,
        q_table_size=svc.agent.get_q_table_size(),
        is_trained=svc.is_trained,
    )


@router.get(
    "/agent/q-table",
    response_model=QTableResponse,
    summary="Get the Q-table (top entries by Q-value)",
)
def get_q_table():
    """
    Return a summary of the Q-table sorted by highest Q-value.
    Limited to top 100 entries to keep the response manageable.
    """
    svc = get_service()
    if svc.agent is None:
        raise HTTPException(status_code=400, detail="Agent not trained yet")

    snapshot = svc.agent.get_q_table_snapshot()
    return QTableResponse(
        size=snapshot["size"],
        top_entries=[QTableEntry(**e) for e in snapshot["top_entries"]],
    )
