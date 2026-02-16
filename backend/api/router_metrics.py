"""
router_metrics.py — API endpoints for training & performance metrics.
=====================================================================
Prefix: /metrics
"""

from fastapi import APIRouter
from backend.models.metrics_models import MetricsSummary, EpisodeMetricPoint
from backend.services.routing_service import get_service

router = APIRouter(prefix="/metrics", tags=["Metrics"])


@router.get(
    "",
    response_model=MetricsSummary,
    summary="Get training & performance metrics",
)
def get_metrics():
    """
    Return aggregate metrics from training and the current network state.
    Includes downsampled reward history suitable for charting.

    **Example response:**
    ```json
    {
      "is_trained": true,
      "total_episodes": 5000,
      "q_table_size": 371,
      "avg_reward_first_500": -12.5,
      "avg_reward_last_500": 67.3,
      "reward_improvement_percent": 638.4,
      "delivery_rate_first_500": 45.0,
      "delivery_rate_last_500": 98.4,
      "reward_history": [
        {"episode": 1, "reward": -34.2, "steps": 50, "delivered": false, "epsilon": 0.995}
      ],
      "network_nodes": 10,
      "network_edges": 38,
      "topology_type": "random"
    }
    ```
    """
    svc = get_service()
    data = svc.get_metrics()

    return MetricsSummary(
        is_trained=data["is_trained"],
        total_episodes=data["total_episodes"],
        q_table_size=data["q_table_size"],
        avg_reward_first_500=data.get("avg_reward_first_500"),
        avg_reward_last_500=data.get("avg_reward_last_500"),
        reward_improvement_percent=data.get("reward_improvement_percent"),
        delivery_rate_first_500=data.get("delivery_rate_first_500"),
        delivery_rate_last_500=data.get("delivery_rate_last_500"),
        reward_history=[EpisodeMetricPoint(**p) for p in data.get("reward_history", [])],
        network_nodes=data.get("network_nodes", 0),
        network_edges=data.get("network_edges", 0),
        topology_type=data.get("topology_type", ""),
    )
