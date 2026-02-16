"""
router_network.py — API endpoints for network topology management.
===================================================================
Prefix: /network
"""

from fastapi import APIRouter, HTTPException
from backend.models.network_models import (
    TopologyCreateRequest, TopologyResponse, NodeResponse, EdgeResponse,
    TopologyMetadata, EdgeUpdateRequest, EdgeUpdateResponse,
    LinkFailureRequest, LinkFailureResponse,
)
from backend.services.routing_service import get_service

router = APIRouter(prefix="/network", tags=["Network"])


@router.get(
    "",
    response_model=TopologyResponse,
    summary="Get current network topology",
    response_description="Full graph with nodes, edges, and metadata",
)
def get_network():
    """
    Retrieve the current network topology with all node and edge attributes.

    **Example response:**
    ```json
    {
      "nodes": [{"id": 0, "label": "Router-0", "neighbor_count": 5}],
      "edges": [{"source": 0, "target": 1, "delay": 3.5, "bandwidth": 45.0,
                  "congestion": 0.12, "loss_rate": 0.01,
                  "congestion_level": "low", "effective_cost": 1.42}],
      "metadata": {"node_count": 10, "edge_count": 38, "topology_type": "random"}
    }
    ```
    """
    svc = get_service()
    data = svc.get_topology()
    return TopologyResponse(
        nodes=[NodeResponse(**n) for n in data["nodes"]],
        edges=[EdgeResponse(**e) for e in data["edges"]],
        metadata=TopologyMetadata(**data["metadata"]),
    )


@router.post(
    "",
    response_model=TopologyResponse,
    status_code=201,
    summary="Create / reset network topology",
)
def create_network(req: TopologyCreateRequest):
    """
    Rebuild the network with the specified topology.
    **Warning:** This discards any trained Q-table.

    **Example request:**
    ```json
    {"topology_type": "random", "num_nodes": 10, "seed": 42}
    ```
    """
    svc = get_service()
    data = svc.reset_network(
        topology_type=req.topology_type.value,
        num_nodes=req.num_nodes,
        seed=req.seed,
    )
    return TopologyResponse(
        nodes=[NodeResponse(**n) for n in data["nodes"]],
        edges=[EdgeResponse(**e) for e in data["edges"]],
        metadata=TopologyMetadata(**data["metadata"]),
    )


@router.put(
    "/edge",
    response_model=EdgeUpdateResponse,
    summary="Update an edge's attributes",
)
def update_edge(req: EdgeUpdateRequest):
    """
    Dynamically update delay, bandwidth, congestion, or loss_rate on a specific edge.
    Useful for simulating congestion spikes.

    **Example request:**
    ```json
    {"source": 0, "target": 1, "congestion": 0.9}
    ```
    """
    svc = get_service()
    try:
        result = svc.update_edge(
            source=req.source,
            target=req.target,
            delay=req.delay,
            bandwidth=req.bandwidth,
            congestion=req.congestion,
            loss_rate=req.loss_rate,
        )
        return EdgeUpdateResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post(
    "/failure",
    response_model=LinkFailureResponse,
    summary="Inject or restore a link failure",
)
def link_failure(req: LinkFailureRequest):
    """
    Simulate link failure (`action: "fail"`) or restoration (`action: "restore"`).

    **Example request:**
    ```json
    {"source": 2, "target": 5, "action": "fail"}
    ```
    """
    svc = get_service()
    result = svc.inject_failure(req.source, req.target, req.action)
    return LinkFailureResponse(**result)
