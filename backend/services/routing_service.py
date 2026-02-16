"""
routing_service.py — Service layer bridging API ↔ Core engine.
===============================================================
Holds the singleton instances of NetworkSimulator and QLearningAgent,
orchestrates training, routing, and metrics collection.

Design: The service is a plain class (not a FastAPI dependency) so it can
be imported and used by all routers.  A module-level singleton is created
at import time; routers get it via `get_service()`.
"""

import time
import random
import logging
import statistics
from typing import Optional

from backend.core.network import NetworkSimulator, TopologyConfig
from backend.core.agent import QLearningAgent, HyperParameters, RoutingResult

logger = logging.getLogger(__name__)


class RoutingService:
    """
    Central service that owns the network and agent instances.

    All mutations (train, topology reset, edge updates) go through this
    class so state stays consistent.
    """

    def __init__(self):
        # Initialise with a sensible default topology
        self._config = TopologyConfig(num_nodes=10, topology_type="random", seed=42)
        self.network = NetworkSimulator(self._config)
        self.agent: Optional[QLearningAgent] = None
        self._is_trained = False
        self._last_train_time: float = 0.0
        logger.info("RoutingService initialised with default 10-node random topology")

    # ------------------------------------------------------------------
    # Network management
    # ------------------------------------------------------------------

    def reset_network(self, topology_type: str = "random", num_nodes: int = 10,
                      seed: Optional[int] = None) -> dict:
        """Rebuild the network and discard the old Q-table."""
        self._config = TopologyConfig(
            num_nodes=num_nodes,
            topology_type=topology_type,
            seed=seed,
        )
        self.network = NetworkSimulator(self._config)
        self.agent = None
        self._is_trained = False
        logger.info("Network reset: %d nodes, type=%s", num_nodes, topology_type)
        return self.get_topology()

    def get_topology(self) -> dict:
        """Return full topology data for the API response."""
        raw = self.network.to_dict()

        nodes = []
        for n in raw["nodes"]:
            nid = n["id"]
            nodes.append({
                "id": nid,
                "label": f"Router-{nid}",
                "neighbor_count": len(self.network.get_neighbors(nid)),
            })

        edges = []
        for e in raw["edges"]:
            cong = e["congestion"]
            # Classify congestion into human-readable levels
            if cong < 0.3:
                level = "low"
            elif cong < 0.6:
                level = "medium"
            elif cong < 0.85:
                level = "high"
            else:
                level = "critical"

            edges.append({
                "source": e["source"],
                "target": e["target"],
                "delay": e["delay"],
                "bandwidth": e["bandwidth"],
                "congestion": round(cong, 4),
                "loss_rate": round(e["loss_rate"], 4),
                "congestion_level": level,
                "effective_cost": self.network.get_edge_cost(e["source"], e["target"]),
            })

        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "node_count": self.network.graph.number_of_nodes(),
                "edge_count": self.network.graph.number_of_edges(),
                "topology_type": self._config.topology_type,
            },
        }

    def update_edge(self, source: int, target: int, **kwargs) -> dict:
        """Update individual edge attributes."""
        if not self.network.graph.has_edge(source, target):
            raise ValueError(f"Edge {source}→{target} does not exist")

        edge_data = self.network.graph[source][target]
        updated = {}
        for key in ("delay", "bandwidth", "congestion", "loss_rate"):
            if key in kwargs and kwargs[key] is not None:
                edge_data[key] = kwargs[key]
                updated[key] = kwargs[key]

        return {
            "source": source,
            "target": target,
            "updated_fields": updated,
            "new_effective_cost": self.network.get_edge_cost(source, target),
        }

    def inject_failure(self, source: int, target: int, action: str) -> dict:
        """Inject or restore a link failure."""
        if action == "fail":
            attrs = self.network.inject_link_failure(source, target)
            return {
                "source": source,
                "target": target,
                "status": "removed" if attrs else "not_found",
                "remaining_edges": self.network.graph.number_of_edges(),
            }
        else:  # restore — we can't restore without original attrs, so add defaults
            self.network.restore_link(source, target, {
                "delay": 5.0, "bandwidth": 50.0, "congestion": 0.1, "loss_rate": 0.01
            })
            return {
                "source": source,
                "target": target,
                "status": "restored",
                "remaining_edges": self.network.graph.number_of_edges(),
            }

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_agent(self, episodes: int = 5000, alpha: float = 0.1,
                    gamma: float = 0.95, epsilon: float = 1.0,
                    epsilon_min: float = 0.01, epsilon_decay: float = 0.995,
                    fluctuate_every: int = 50) -> dict:
        """Train (or retrain) the Q-learning agent."""
        hp = HyperParameters(
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
        )
        self.agent = QLearningAgent(self.network, hp)

        t0 = time.time()
        history = self.agent.train(episodes=episodes, fluctuate_every=fluctuate_every)
        elapsed = round(time.time() - t0, 2)
        self._is_trained = True
        self._last_train_time = elapsed

        # Compute summary stats from last 500 episodes
        tail = history[-500:] if len(history) >= 500 else history
        avg_reward = statistics.mean(ep.total_reward for ep in tail)
        delivery_rate = sum(1 for ep in tail if ep.delivered) / len(tail) * 100

        return {
            "status": "completed",
            "episodes_completed": episodes,
            "q_table_size": self.agent.get_q_table_size(),
            "final_epsilon": round(self.agent._epsilon, 6),
            "avg_reward_last_500": round(avg_reward, 2),
            "delivery_rate_last_500": round(delivery_rate, 2),
            "training_time_seconds": elapsed,
        }

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def route_q_learning(self, source: int, destination: int) -> dict:
        """Route a packet using the trained Q-learning policy."""
        if not self._is_trained or self.agent is None:
            raise RuntimeError("Agent not trained yet. Call POST /train first.")

        result: RoutingResult = self.agent.route_packet(source, destination)

        # Build per-hop details
        per_hop = []
        for i in range(len(result.path) - 1):
            u, v = result.path[i], result.path[i + 1]
            per_hop.append({
                "from_node": u,
                "to_node": v,
                "edge_cost": self.network.get_edge_cost(u, v),
                "q_value": round(self.agent._get_q((u, destination), v), 4),
            })

        return {
            "algorithm": "q-learning",
            "source": source,
            "destination": destination,
            "path": result.path,
            "total_cost": result.total_cost,
            "hop_count": result.hop_count,
            "delivered": result.delivered,
            "reason": result.reason,
            "per_hop_details": per_hop,
        }

    def route_dijkstra(self, source: int, destination: int) -> dict:
        """Route a packet using static Dijkstra shortest path."""
        try:
            path, cost = self.network.dijkstra_shortest_path(source, destination)
            per_hop = []
            for i in range(len(path) - 1):
                per_hop.append({
                    "from_node": path[i],
                    "to_node": path[i + 1],
                    "edge_cost": self.network.get_edge_cost(path[i], path[i + 1]),
                    "q_value": None,
                })
            return {
                "algorithm": "dijkstra",
                "source": source,
                "destination": destination,
                "path": path,
                "total_cost": cost,
                "hop_count": len(path) - 1,
                "delivered": True,
                "reason": "delivered",
                "per_hop_details": per_hop,
            }
        except Exception as e:
            return {
                "algorithm": "dijkstra",
                "source": source,
                "destination": destination,
                "path": [],
                "total_cost": 0.0,
                "hop_count": 0,
                "delivered": False,
                "reason": str(e),
                "per_hop_details": [],
            }

    def compare_routes(self, num_packets: int = 50, seed: Optional[int] = None) -> dict:
        """Run both algorithms on the same random packet set and compare."""
        if not self._is_trained or self.agent is None:
            raise RuntimeError("Agent not trained yet. Call POST /train first.")

        rng = random.Random(seed)
        nodes = list(self.network.graph.nodes())
        pairs = [(rng.choice(nodes), rng.choice(nodes)) for _ in range(num_packets)]
        # Avoid src == dst
        pairs = [(s, d) for s, d in pairs if s != d]

        ql_costs, ql_hops, ql_delivered = [], [], 0
        dj_costs, dj_hops, dj_delivered = [], [], 0

        for src, dst in pairs:
            ql = self.route_q_learning(src, dst)
            dj = self.route_dijkstra(src, dst)

            if ql["delivered"]:
                ql_costs.append(ql["total_cost"])
                ql_hops.append(ql["hop_count"])
                ql_delivered += 1

            if dj["delivered"]:
                dj_costs.append(dj["total_cost"])
                dj_hops.append(dj["hop_count"])
                dj_delivered += 1

        n = len(pairs)
        ql_avg_cost = statistics.mean(ql_costs) if ql_costs else 0
        dj_avg_cost = statistics.mean(dj_costs) if dj_costs else 0
        ql_avg_hops = statistics.mean(ql_hops) if ql_hops else 0
        dj_avg_hops = statistics.mean(dj_hops) if dj_hops else 0
        ql_dr = ql_delivered / n * 100 if n else 0
        dj_dr = dj_delivered / n * 100 if n else 0

        cost_diff = ((ql_avg_cost - dj_avg_cost) / dj_avg_cost * 100) if dj_avg_cost else 0
        hop_diff = ((ql_avg_hops - dj_avg_hops) / dj_avg_hops * 100) if dj_avg_hops else 0

        return {
            "num_packets": n,
            "q_learning": {
                "avg_cost": round(ql_avg_cost, 4),
                "avg_hops": round(ql_avg_hops, 2),
                "delivery_rate": round(ql_dr, 2),
                "min_cost": round(min(ql_costs, default=0), 4),
                "max_cost": round(max(ql_costs, default=0), 4),
            },
            "dijkstra": {
                "avg_cost": round(dj_avg_cost, 4),
                "avg_hops": round(dj_avg_hops, 2),
                "delivery_rate": round(dj_dr, 2),
                "min_cost": round(min(dj_costs, default=0), 4),
                "max_cost": round(max(dj_costs, default=0), 4),
            },
            "delta": {
                "cost_diff_percent": round(cost_diff, 2),
                "hop_diff_percent": round(hop_diff, 2),
                "delivery_diff": round(ql_dr - dj_dr, 2),
                "winner_by_delivery": "q-learning" if ql_dr >= dj_dr else "dijkstra",
                "winner_by_cost": "q-learning" if ql_avg_cost <= dj_avg_cost else "dijkstra",
            },
        }

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_metrics(self) -> dict:
        """Return aggregate metrics for the /metrics endpoint."""
        if not self._is_trained or self.agent is None:
            return {
                "is_trained": False,
                "total_episodes": 0,
                "q_table_size": 0,
                "network_nodes": self.network.graph.number_of_nodes(),
                "network_edges": self.network.graph.number_of_edges(),
                "topology_type": self._config.topology_type,
            }

        history = self.agent.training_history
        total = len(history)

        first_500 = history[:500] if total >= 500 else history
        last_500 = history[-500:] if total >= 500 else history

        avg_first = statistics.mean(ep.total_reward for ep in first_500)
        avg_last = statistics.mean(ep.total_reward for ep in last_500)
        dr_first = sum(1 for ep in first_500 if ep.delivered) / len(first_500) * 100
        dr_last = sum(1 for ep in last_500 if ep.delivered) / len(last_500) * 100

        improvement = ((avg_last - avg_first) / abs(avg_first) * 100) if avg_first != 0 else 0

        # Downsample reward history for charting (max 200 points)
        step = max(1, total // 200)
        reward_history = [
            {
                "episode": ep.episode,
                "reward": round(ep.total_reward, 2),
                "steps": ep.steps,
                "delivered": ep.delivered,
                "epsilon": ep.epsilon,
            }
            for ep in history[::step]
        ]

        return {
            "is_trained": True,
            "total_episodes": total,
            "q_table_size": self.agent.get_q_table_size(),
            "avg_reward_first_500": round(avg_first, 2),
            "avg_reward_last_500": round(avg_last, 2),
            "reward_improvement_percent": round(improvement, 2),
            "delivery_rate_first_500": round(dr_first, 2),
            "delivery_rate_last_500": round(dr_last, 2),
            "reward_history": reward_history,
            "network_nodes": self.network.graph.number_of_nodes(),
            "network_edges": self.network.graph.number_of_edges(),
            "topology_type": self._config.topology_type,
        }

    @property
    def is_trained(self) -> bool:
        return self._is_trained


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_service_instance: Optional[RoutingService] = None


def get_service() -> RoutingService:
    """Return the module-level singleton RoutingService."""
    global _service_instance
    if _service_instance is None:
        _service_instance = RoutingService()
    return _service_instance
