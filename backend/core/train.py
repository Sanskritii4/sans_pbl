"""
train.py — Training Script & Performance Comparison
=====================================================
Entry point that:
    1. Builds a network topology.
    2. Trains the Q-learning agent.
    3. Compares Q-learning routes against Dijkstra shortest paths.
    4. Plots the learning curve (reward over episodes).
    5. Prints a detailed comparison table.

Usage:
    python -m backend.core.train          (from project root)
    python backend/core/train.py          (direct execution)
"""

import sys
import os
import random
import logging
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — safe for servers / CI
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Handle imports for both `python -m backend.core.train` and direct execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Add project root to sys.path so `backend.core` is importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.core.network import NetworkSimulator, TopologyConfig
from backend.core.agent import QLearningAgent, HyperParameters

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TRAINING_EPISODES = 5000
NUM_TEST_PACKETS = 100
TOPOLOGY = "random"        # "mesh" | "grid" | "random"
NUM_NODES = 10
SEED = 42
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "experiments"


# ---------------------------------------------------------------------------
# Utility: moving average for smooth learning curves
# ---------------------------------------------------------------------------

def moving_average(values: list[float], window: int = 100) -> list[float]:
    """Compute a simple moving average for a list of values."""
    if len(values) < window:
        window = max(1, len(values))
    cumsum = np.cumsum(values)
    cumsum = np.insert(cumsum, 0, 0)
    return ((cumsum[window:] - cumsum[:-window]) / window).tolist()


# ---------------------------------------------------------------------------
# Plot: learning curve
# ---------------------------------------------------------------------------

def plot_learning_curve(history, output_path: Path) -> None:
    """
    Plot two charts:
      1. Episode reward (raw + smoothed moving average)
      2. Delivery rate over time (moving window)
    """
    rewards = [ep.total_reward for ep in history]
    delivered = [1 if ep.delivered else 0 for ep in history]
    episodes = list(range(1, len(history) + 1))

    smoothed_rewards = moving_average(rewards, window=100)
    smoothed_delivery = moving_average(delivered, window=100)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), dpi=120)

    # --- Reward plot ---
    ax1 = axes[0]
    ax1.plot(episodes, rewards, alpha=0.15, color="#636EFA", linewidth=0.5, label="Raw reward")
    ax1.plot(
        episodes[99:] if len(episodes) > 99 else episodes,
        smoothed_rewards,
        color="#EF553B", linewidth=2, label="Moving avg (100 ep)"
    )
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.set_title("Q-Learning Training Curve — Reward per Episode")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Delivery rate plot ---
    ax2 = axes[1]
    ax2.plot(
        episodes[99:] if len(episodes) > 99 else episodes,
        [d * 100 for d in smoothed_delivery],
        color="#00CC96", linewidth=2
    )
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Delivery Rate (%)")
    ax2.set_title("Packet Delivery Rate Over Training")
    ax2.set_ylim(-5, 105)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    logger.info("Learning curve saved → %s", output_path)


# ---------------------------------------------------------------------------
# Comparison: Q-Learning vs Dijkstra
# ---------------------------------------------------------------------------

def compare_algorithms(agent: QLearningAgent, network: NetworkSimulator, num_packets: int):
    """
    Route `num_packets` random (src, dst) pairs with both Q-learning and
    Dijkstra, then print a side-by-side comparison table.
    """
    nodes = list(network.graph.nodes())
    results_ql = []
    results_dj = []

    # Use the same (src, dst) pairs for both algorithms for fair comparison
    pairs = [tuple(random.sample(nodes, 2)) for _ in range(num_packets)]

    for src, dst in pairs:
        # --- Q-Learning ---
        ql_result = agent.route_packet(src, dst)
        results_ql.append(ql_result)

        # --- Dijkstra ---
        try:
            dj_path, dj_cost = network.dijkstra_shortest_path(src, dst)
            results_dj.append({
                "path": dj_path,
                "cost": dj_cost,
                "hops": len(dj_path) - 1,
                "delivered": True,
            })
        except Exception:
            results_dj.append({
                "path": [],
                "cost": float("inf"),
                "hops": 0,
                "delivered": False,
            })

    # --------------- Aggregate metrics ---------------

    ql_costs = [r.total_cost for r in results_ql if r.delivered]
    dj_costs = [r["cost"] for r in results_dj if r["delivered"]]

    ql_hops = [r.hop_count for r in results_ql if r.delivered]
    dj_hops = [r["hops"] for r in results_dj if r["delivered"]]

    ql_delivery = sum(1 for r in results_ql if r.delivered) / num_packets * 100
    dj_delivery = sum(1 for r in results_dj if r["delivered"]) / num_packets * 100

    ql_avg_cost = statistics.mean(ql_costs) if ql_costs else float("inf")
    dj_avg_cost = statistics.mean(dj_costs) if dj_costs else float("inf")

    ql_avg_hops = statistics.mean(ql_hops) if ql_hops else 0
    dj_avg_hops = statistics.mean(dj_hops) if dj_hops else 0

    # --------------- Print comparison ---------------

    header = f"\n{'='*65}"
    print(header)
    print(f"  PERFORMANCE COMPARISON — Q-Learning vs Dijkstra")
    print(f"  {num_packets} packets | {network.config.num_nodes}-node {network.config.topology_type} topology")
    print(header)
    print(f"  {'Metric':<30} {'Q-Learning':>14} {'Dijkstra':>14}")
    print(f"  {'-'*30} {'-'*14} {'-'*14}")
    print(f"  {'Avg Path Cost':<30} {ql_avg_cost:>14.4f} {dj_avg_cost:>14.4f}")
    print(f"  {'Avg Hop Count':<30} {ql_avg_hops:>14.2f} {dj_avg_hops:>14.2f}")
    print(f"  {'Delivery Rate (%)':<30} {ql_delivery:>13.1f}% {dj_delivery:>13.1f}%")
    if ql_costs:
        print(f"  {'Cost Std Dev':<30} {statistics.stdev(ql_costs) if len(ql_costs)>1 else 0:>14.4f} {statistics.stdev(dj_costs) if len(dj_costs)>1 else 0:>14.4f}")
    print(header)

    # Winner analysis
    if ql_delivery > dj_delivery:
        print("  ✅ Q-Learning wins on DELIVERY RATE (adapts to congestion/failures)")
    elif dj_delivery > ql_delivery:
        print("  ✅ Dijkstra wins on DELIVERY RATE (static but reliable)")
    else:
        print("  🤝 Tied on delivery rate")

    if ql_avg_cost < dj_avg_cost:
        print("  ✅ Q-Learning wins on PATH COST")
    elif dj_avg_cost < ql_avg_cost:
        print("  ✅ Dijkstra wins on PATH COST (expected — it's optimal for static weights)")
    else:
        print("  🤝 Tied on path cost")

    print()

    return {
        "q_learning": {
            "avg_cost": round(ql_avg_cost, 4),
            "avg_hops": round(ql_avg_hops, 2),
            "delivery_rate": round(ql_delivery, 2),
        },
        "dijkstra": {
            "avg_cost": round(dj_avg_cost, 4),
            "avg_hops": round(dj_avg_hops, 2),
            "delivery_rate": round(dj_delivery, 2),
        },
    }


# ---------------------------------------------------------------------------
# Stress test: inject failures and re-compare
# ---------------------------------------------------------------------------

def stress_test_with_failures(agent: QLearningAgent, network: NetworkSimulator, num_packets: int = 50):
    """
    Inject random link failures and congestion spikes, then compare again.
    This is where Q-learning should shine — Dijkstra uses stale static weights.
    """
    print("\n" + "=" * 65)
    print("  STRESS TEST — Dynamic Network Conditions")
    print("=" * 65)

    edges = list(network.graph.edges())
    num_failures = max(1, len(edges) // 10)  # fail ~10% of links

    # Inject failures
    saved_links = []
    for _ in range(num_failures):
        if edges:
            u, v = random.choice(edges)
            attrs = network.inject_link_failure(u, v)
            if attrs:
                saved_links.append((u, v, attrs))
                edges = list(network.graph.edges())  # refresh

    # Spike congestion on random links
    for u, v in random.sample(list(network.graph.edges()), min(5, network.graph.number_of_edges())):
        network.inject_congestion_spike(u, v, level=random.uniform(0.7, 1.0))

    print(f"  Injected {len(saved_links)} link failures + 5 congestion spikes\n")

    # Re-compare under stress
    compare_algorithms(agent, network, num_packets)

    # Restore links
    for u, v, attrs in saved_links:
        network.restore_link(u, v, attrs)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    random.seed(SEED)

    print("\n" + "=" * 65)
    print("  AI-Driven Adaptive Packet Routing — Q-Learning Trainer")
    print("=" * 65)

    # 1. Build network
    config = TopologyConfig(
        num_nodes=NUM_NODES,
        topology_type=TOPOLOGY,
        seed=SEED,
    )
    network = NetworkSimulator(config)
    print(f"\n  Network: {NUM_NODES} nodes, {network.graph.number_of_edges()} edges, type={TOPOLOGY}")

    # 2. Initialise Q-learning agent
    hp = HyperParameters(
        alpha=0.1,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        max_steps_per_episode=50,
    )
    agent = QLearningAgent(network, hp)

    # 3. Train
    print(f"\n  Training for {TRAINING_EPISODES} episodes...")
    print(f"  Hyperparameters: α={hp.alpha}, γ={hp.gamma}, ε₀={hp.epsilon}, decay={hp.epsilon_decay}\n")

    history = agent.train(episodes=TRAINING_EPISODES)

    # 4. Plot learning curve
    plot_path = OUTPUT_DIR / "learning_curve.png"
    plot_learning_curve(history, plot_path)
    print(f"  📊 Learning curve saved → {plot_path}")

    # 5. Compare with Dijkstra under normal conditions
    compare_algorithms(agent, network, NUM_TEST_PACKETS)

    # 6. Stress test with dynamic failures
    stress_test_with_failures(agent, network, num_packets=50)

    # 7. Summary stats
    print(f"  Q-table size: {agent.get_q_table_size()} entries")
    print(f"  Final ε: {agent._epsilon:.6f}")
    print()


if __name__ == "__main__":
    main()
