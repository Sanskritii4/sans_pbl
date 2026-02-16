"""
network.py — Network Simulation Module
========================================
Simulates a packet-switched network using NetworkX.
Each edge has: delay (ms), bandwidth (Mbps), congestion (0.0–1.0), loss_rate (0.0–1.0).
Supports dynamic congestion fluctuation and random link failure injection.
"""

import random
import logging
import networkx as nx
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

# ---------------------------------------------------------------------------
# Data classes for type-safe edge / topology metadata
# ---------------------------------------------------------------------------

@dataclass
class EdgeAttributes:
    """Immutable snapshot of a single edge's properties."""
    delay: float          # propagation delay in ms
    bandwidth: float      # link capacity in Mbps
    congestion: float     # current congestion level ∈ [0, 1]
    loss_rate: float      # packet loss probability ∈ [0, 1]


@dataclass
class TopologyConfig:
    """Configuration for building a network topology."""
    num_nodes: int = 10
    topology_type: str = "mesh"           # "mesh" | "grid" | "random"
    min_delay: float = 1.0                # ms
    max_delay: float = 10.0
    min_bandwidth: float = 10.0           # Mbps
    max_bandwidth: float = 100.0
    initial_congestion: float = 0.1
    initial_loss_rate: float = 0.01
    random_edge_prob: float = 0.4         # for random topology (Erdős–Rényi)
    seed: Optional[int] = None


# ---------------------------------------------------------------------------
# NetworkSimulator — main class
# ---------------------------------------------------------------------------

class NetworkSimulator:
    """
    Wraps a NetworkX DiGraph to simulate a packet network.

    Design decisions:
    - DiGraph (directed) because real links can be asymmetric in delay/bandwidth.
    - Edge attributes stored directly on the NetworkX edge dict for O(1) access.
    - Congestion is modeled as a float ∈ [0, 1] that scales effective delay.
    - Effective cost of traversing an edge:
          cost = delay * (1 + congestion) + loss_penalty
      This single scalar is what the RL agent optimises against.
    """

    def __init__(self, config: Optional[TopologyConfig] = None):
        self.config = config or TopologyConfig()
        self.graph: nx.DiGraph = nx.DiGraph()
        self._rng = random.Random(self.config.seed)
        self._np_rng = np.random.default_rng(self.config.seed)
        self._build_topology()
        logger.info(
            "Network initialised: %d nodes, %d edges, type=%s",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
            self.config.topology_type,
        )

    # ------------------------------------------------------------------
    # Topology construction
    # ------------------------------------------------------------------

    def _build_topology(self) -> None:
        """Dispatch to the appropriate topology builder."""
        builders = {
            "mesh": self._build_mesh,
            "grid": self._build_grid,
            "random": self._build_random,
        }
        builder = builders.get(self.config.topology_type)
        if builder is None:
            raise ValueError(f"Unknown topology type: {self.config.topology_type}")
        builder()

    def _random_edge_attrs(self) -> dict:
        """Generate random edge attributes within configured bounds."""
        return {
            "delay": round(self._rng.uniform(self.config.min_delay, self.config.max_delay), 2),
            "bandwidth": round(self._rng.uniform(self.config.min_bandwidth, self.config.max_bandwidth), 2),
            "congestion": self.config.initial_congestion,
            "loss_rate": self.config.initial_loss_rate,
        }

    def _build_mesh(self) -> None:
        """
        Full-mesh topology: every node connects to every other node.
        Good for small networks (≤12 nodes) to give the agent maximum routing freedom.
        """
        n = self.config.num_nodes
        self.graph.add_nodes_from(range(n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    self.graph.add_edge(i, j, **self._random_edge_attrs())

    def _build_grid(self) -> None:
        """
        2D grid topology.  Nodes arranged in a ⌈√n⌉ × ⌈√n⌉ grid with
        bidirectional links to 4-connected neighbours.
        """
        n = self.config.num_nodes
        side = int(np.ceil(np.sqrt(n)))
        self.graph.add_nodes_from(range(n))
        for idx in range(n):
            r, c = divmod(idx, side)
            # right neighbour
            right = idx + 1
            if c + 1 < side and right < n:
                self.graph.add_edge(idx, right, **self._random_edge_attrs())
                self.graph.add_edge(right, idx, **self._random_edge_attrs())
            # bottom neighbour
            down = idx + side
            if down < n:
                self.graph.add_edge(idx, down, **self._random_edge_attrs())
                self.graph.add_edge(down, idx, **self._random_edge_attrs())

    def _build_random(self) -> None:
        """
        Erdős–Rényi random graph.  Each directed edge exists independently
        with probability `random_edge_prob`.  We ensure the graph is weakly
        connected by adding a spanning-tree backbone first.
        """
        n = self.config.num_nodes
        self.graph.add_nodes_from(range(n))
        # Backbone: random spanning tree to guarantee connectivity
        nodes = list(range(n))
        self._rng.shuffle(nodes)
        for i in range(len(nodes) - 1):
            self.graph.add_edge(nodes[i], nodes[i + 1], **self._random_edge_attrs())
            self.graph.add_edge(nodes[i + 1], nodes[i], **self._random_edge_attrs())
        # Additional random edges
        for i in range(n):
            for j in range(n):
                if i != j and not self.graph.has_edge(i, j):
                    if self._rng.random() < self.config.random_edge_prob:
                        self.graph.add_edge(i, j, **self._random_edge_attrs())

    # ------------------------------------------------------------------
    # Edge queries
    # ------------------------------------------------------------------

    def get_edge_attrs(self, src: int, dst: int) -> EdgeAttributes:
        """Return a typed snapshot of edge attributes."""
        d = self.graph[src][dst]
        return EdgeAttributes(
            delay=d["delay"],
            bandwidth=d["bandwidth"],
            congestion=d["congestion"],
            loss_rate=d["loss_rate"],
        )

    def get_edge_cost(self, src: int, dst: int) -> float:
        """
        Compute effective traversal cost for the RL agent's reward function.

        cost = delay × (1 + congestion) + 50 × loss_rate − 0.1 × bandwidth

        Rationale:
        - delay × (1 + congestion):  congestion amplifies latency linearly.
        - 50 × loss_rate:            heavy penalty for lossy links (50 ms equiv.).
        - −0.1 × bandwidth:          small bonus for high-bandwidth links (throughput).
        """
        d = self.graph[src][dst]
        raw_cost = (
            d["delay"] * (1.0 + d["congestion"])
            + 50.0 * d["loss_rate"]
            - 0.1 * d["bandwidth"]
        )
        # Floor at 0.1 — Dijkstra requires non-negative edge weights.
        # Negative costs occur when bandwidth bonus exceeds delay on fast links.
        return round(max(0.1, raw_cost), 4)

    def get_neighbors(self, node: int) -> list[int]:
        """Return list of nodes reachable from `node` in one hop."""
        return list(self.graph.successors(node))

    # ------------------------------------------------------------------
    # Dynamic network changes (called between / during episodes)
    # ------------------------------------------------------------------

    def fluctuate_congestion(
        self, intensity: float = 0.15, clip_low: float = 0.0, clip_high: float = 1.0
    ) -> None:
        """
        Randomly perturb every edge's congestion by ±intensity (Gaussian noise).
        Simulates real-world traffic fluctuations between routing decisions.
        """
        for u, v, d in self.graph.edges(data=True):
            delta = self._np_rng.normal(0, intensity)
            d["congestion"] = float(np.clip(d["congestion"] + delta, clip_low, clip_high))

    def fluctuate_loss(
        self, intensity: float = 0.02, clip_low: float = 0.0, clip_high: float = 0.3
    ) -> None:
        """Randomly perturb packet loss rates."""
        for u, v, d in self.graph.edges(data=True):
            delta = self._np_rng.normal(0, intensity)
            d["loss_rate"] = float(np.clip(d["loss_rate"] + delta, clip_low, clip_high))

    def inject_congestion_spike(self, src: int, dst: int, level: float = 0.9) -> None:
        """Set a specific edge to high congestion (simulate bottleneck)."""
        if self.graph.has_edge(src, dst):
            self.graph[src][dst]["congestion"] = level
            logger.info("Congestion spike injected on edge %d→%d (%.2f)", src, dst, level)

    def inject_link_failure(self, src: int, dst: int) -> Optional[dict]:
        """
        Remove an edge entirely.  Returns the old attributes so the link
        can be restored later, or None if the edge didn't exist.
        """
        if self.graph.has_edge(src, dst):
            attrs = dict(self.graph[src][dst])
            self.graph.remove_edge(src, dst)
            logger.info("Link failure injected: %d→%d removed", src, dst)
            return attrs
        return None

    def restore_link(self, src: int, dst: int, attrs: dict) -> None:
        """Restore a previously failed link with its original attributes."""
        self.graph.add_edge(src, dst, **attrs)
        logger.info("Link restored: %d→%d", src, dst)

    # ------------------------------------------------------------------
    # Dijkstra baseline
    # ------------------------------------------------------------------

    def dijkstra_shortest_path(self, src: int, dst: int) -> tuple[list[int], float]:
        """
        Compute shortest path using Dijkstra with `get_edge_cost` as weight.
        Returns (path, total_cost).  Raises nx.NetworkXNoPath if unreachable.
        """
        # Build a weight function that uses our composite cost metric
        path = nx.dijkstra_path(self.graph, src, dst, weight=lambda u, v, d: self.get_edge_cost(u, v))
        cost = sum(self.get_edge_cost(path[i], path[i + 1]) for i in range(len(path) - 1))
        return path, round(cost, 4)

    # ------------------------------------------------------------------
    # Serialisation (for API / frontend)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise full topology for JSON transport."""
        nodes = [{"id": n} for n in self.graph.nodes()]
        edges = []
        for u, v, d in self.graph.edges(data=True):
            edges.append({
                "source": u,
                "target": v,
                "delay": d["delay"],
                "bandwidth": d["bandwidth"],
                "congestion": round(d["congestion"], 4),
                "loss_rate": round(d["loss_rate"], 4),
            })
        return {"nodes": nodes, "edges": edges}
