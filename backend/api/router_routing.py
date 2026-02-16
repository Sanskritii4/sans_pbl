"""
router_routing.py — API endpoints for packet routing.
======================================================
Prefix: /route & /static-route
"""

from fastapi import APIRouter, HTTPException, Query
from backend.models.routing_models import (
    RouteResponse, HopDetail, CompareRequest, CompareResponse,
    AlgorithmSummary, ComparisonDelta,
)
from backend.services.routing_service import get_service

router = APIRouter(tags=["Routing"])


@router.get(
    "/route",
    response_model=RouteResponse,
    summary="Route a packet using Q-learning",
)
def route_q_learning(
    source: int = Query(..., ge=0, description="Source node ID"),
    dest: int = Query(..., ge=0, description="Destination node ID"),
):
    """
    Route a single packet from `source` to `dest` using the trained Q-learning policy.

    **Example:** `GET /route?source=0&dest=9`

    **Example response:**
    ```json
    {
      "algorithm": "q-learning",
      "source": 0, "destination": 9,
      "path": [0, 3, 6, 9],
      "total_cost": 12.34,
      "hop_count": 3,
      "delivered": true,
      "reason": "delivered",
      "per_hop_details": [
        {"from_node": 0, "to_node": 3, "edge_cost": 4.12, "q_value": 73.2}
      ]
    }
    ```
    """
    svc = get_service()
    _validate_nodes(svc, source, dest)

    try:
        result = svc.route_q_learning(source, dest)
        return RouteResponse(
            **{**result, "per_hop_details": [HopDetail(**h) for h in result["per_hop_details"]]}
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get(
    "/static-route",
    response_model=RouteResponse,
    summary="Route a packet using Dijkstra (static baseline)",
)
def route_dijkstra(
    source: int = Query(..., ge=0, description="Source node ID"),
    dest: int = Query(..., ge=0, description="Destination node ID"),
):
    """
    Route a single packet from `source` to `dest` using Dijkstra's shortest-path algorithm.
    This acts as the static baseline for comparison.

    **Example:** `GET /static-route?source=0&dest=9`
    """
    svc = get_service()
    _validate_nodes(svc, source, dest)

    result = svc.route_dijkstra(source, dest)
    return RouteResponse(
        **{**result, "per_hop_details": [HopDetail(**h) for h in result["per_hop_details"]]}
    )


@router.post(
    "/route/compare",
    response_model=CompareResponse,
    summary="Batch compare Q-Learning vs Dijkstra",
)
def compare_routes(req: CompareRequest):
    """
    Route `num_packets` random packets with both algorithms and return
    aggregate comparison metrics.

    **Example request:**
    ```json
    {"num_packets": 50, "seed": 42}
    ```

    **Example response:**
    ```json
    {
      "num_packets": 50,
      "q_learning": {"avg_cost": 15.3, "avg_hops": 4.2, "delivery_rate": 100.0, ...},
      "dijkstra":   {"avg_cost": 14.1, "avg_hops": 3.8, "delivery_rate": 26.0, ...},
      "delta":      {"cost_diff_percent": 8.5, "winner_by_delivery": "q-learning", ...}
    }
    ```
    """
    svc = get_service()
    try:
        result = svc.compare_routes(num_packets=req.num_packets, seed=req.seed)
        return CompareResponse(
            num_packets=result["num_packets"],
            q_learning=AlgorithmSummary(**result["q_learning"]),
            dijkstra=AlgorithmSummary(**result["dijkstra"]),
            delta=ComparisonDelta(**result["delta"]),
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_nodes(svc, source: int, dest: int):
    """Raise 404 if source or dest are not valid node IDs."""
    nodes = set(svc.network.graph.nodes())
    if source not in nodes:
        raise HTTPException(status_code=404, detail=f"Source node {source} not found in network")
    if dest not in nodes:
        raise HTTPException(status_code=404, detail=f"Destination node {dest} not found in network")
    if source == dest:
        raise HTTPException(status_code=400, detail="Source and destination must be different")
