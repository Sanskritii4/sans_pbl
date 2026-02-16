"""
agent.py — Q-Learning Agent for Adaptive Packet Routing
=========================================================
Implements a tabular Q-learning agent that learns to route packets through
a simulated network by interacting with the NetworkSimulator environment.

Key design decisions:
- State  = (current_node, destination_node)
    Kept simple for tabular Q-learning; congestion is captured implicitly
    through the reward signal (which reads live edge attributes).
- Action = next_hop_node  (chosen from current node's neighbours)
- Q-table stored as a flat dict[(state, action) → float] for O(1) access.
- Epsilon-greedy exploration with exponential decay.
"""

import random
import logging
from dataclasses import dataclass, field
from typing import Optional

from .network import NetworkSimulator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases & result containers
# ---------------------------------------------------------------------------

State = tuple[int, int]  # (current_node, destination_node)


@dataclass
class RoutingResult:
    """Outcome of routing a single packet."""
    source: int
    destination: int
    path: list[int]
    total_cost: float
    hop_count: int
    delivered: bool
    reason: str = ""       # "delivered" | "loop" | "no_neighbors" | "max_steps"


@dataclass
class EpisodeStats:
    """Metrics from a single training episode."""
    episode: int
    total_reward: float
    steps: int
    path: list[int]
    delivered: bool
    epsilon: float


@dataclass
class HyperParameters:
    """All tuneable knobs in one place for easy experimentation."""
    alpha: float = 0.1              # learning rate
    gamma: float = 0.95             # discount factor
    epsilon: float = 1.0            # initial exploration rate
    epsilon_min: float = 0.01       # floor for ε
    epsilon_decay: float = 0.995    # multiplicative decay per episode
    max_steps_per_episode: int = 50 # prevent infinite loops during training


# ---------------------------------------------------------------------------
# QLearningAgent
# ---------------------------------------------------------------------------

class QLearningAgent:
    """
    Tabular Q-learning agent for packet routing.

    Training workflow:
        1. For each episode, pick a random (src, dst) pair.
        2. Walk the network using ε-greedy, collecting rewards.
        3. Update Q-values via the Bellman equation after every step.
        4. Decay ε after each episode.

    After training, call `route_packet(src, dst)` with ε = 0 to
    exploit the learned policy.
    """

    def __init__(
        self,
        network: NetworkSimulator,
        hyperparams: Optional[HyperParameters] = None,
    ):
        self.network = network
        self.hp = hyperparams or HyperParameters()

        # Q-table: maps (state, action) → expected cumulative reward
        # defaulting to 0.0 for unseen (state, action) pairs (optimistic init)
        self.q_table: dict[tuple[State, int], float] = {}

        # Internal copy of epsilon so we can decay it during training
        self._epsilon = self.hp.epsilon

        # Training history for plotting learning curves
        self.training_history: list[EpisodeStats] = []

        logger.info("Q-Agent initialised | α=%.3f  γ=%.3f  ε₀=%.2f  decay=%.4f",
                     self.hp.alpha, self.hp.gamma, self.hp.epsilon, self.hp.epsilon_decay)

    # ------------------------------------------------------------------
    # Q-table access helpers
    # ------------------------------------------------------------------

    def _get_q(self, state: State, action: int) -> float:
        """Retrieve Q-value; default to 0.0 for unseen pairs."""
        return self.q_table.get((state, action), 0.0)

    def _set_q(self, state: State, action: int, value: float) -> None:
        self.q_table[(state, action)] = value

    def _max_q(self, state: State, actions: list[int]) -> float:
        """max_a Q(s', a') — used in the Bellman target."""
        if not actions:
            return 0.0
        return max(self._get_q(state, a) for a in actions)

    # ------------------------------------------------------------------
    # Policy
    # ------------------------------------------------------------------

    def select_action(self, state: State, valid_actions: list[int], exploit_only: bool = False) -> int:
        """
        ε-greedy action selection.

        - With probability ε  → pick a random neighbour  (EXPLORE)
        - With probability 1−ε → pick the action with highest Q-value (EXPLOIT)
        - When `exploit_only=True`, always exploit (used at inference time).
        """
        if not valid_actions:
            raise ValueError(f"No valid actions from state {state}")

        if not exploit_only and random.random() < self._epsilon:
            return random.choice(valid_actions)

        # Exploit: argmax Q(s, a) over valid actions
        q_values = {a: self._get_q(state, a) for a in valid_actions}
        max_q = max(q_values.values())
        # Break ties randomly among actions sharing the max Q-value
        best_actions = [a for a, q in q_values.items() if q == max_q]
        return random.choice(best_actions)

    # ------------------------------------------------------------------
    # Reward function
    # ------------------------------------------------------------------

    def compute_reward(
        self,
        current: int,
        next_hop: int,
        destination: int,
        visited: set[int],
    ) -> float:
        """
        Multi-factor reward signal.

        Components:
        ┌───────────────────┬───────────┬────────────────────────────────────┐
        │ Factor            │ Sign      │ Rationale                          │
        ├───────────────────┼───────────┼────────────────────────────────────┤
        │ Goal reached      │  +100     │ Strong terminal incentive          │
        │ Loop detected     │  −50      │ Penalise revisiting nodes          │
        │ Delay penalty     │  −delay   │ Prefer low-latency links           │
        │ Congestion penalty│  −20×cong │ Avoid congested links              │
        │ Loss penalty      │  −50×loss │ Avoid lossy links                  │
        │ Throughput bonus   │  +0.1×bw  │ Prefer high-bandwidth links        │
        └───────────────────┴───────────┴────────────────────────────────────┘
        """
        # Terminal reward: packet arrived
        if next_hop == destination:
            return 100.0

        # Loop penalty: already visited this node
        if next_hop in visited:
            return -50.0

        # Retrieve live edge attributes (reflect current congestion / loss)
        attrs = self.network.get_edge_attrs(current, next_hop)

        reward = 0.0
        reward -= attrs.delay                     # delay penalty (negative)
        reward -= 20.0 * attrs.congestion         # congestion penalty
        reward -= 50.0 * attrs.loss_rate          # packet loss penalty
        reward += 0.1 * attrs.bandwidth           # throughput bonus (small)
        reward -= 1.0                             # per-hop cost to minimise path length

        return round(reward, 4)

    # ------------------------------------------------------------------
    # Bellman update
    # ------------------------------------------------------------------

    def update(
        self,
        state: State,
        action: int,
        reward: float,
        next_state: State,
        next_valid_actions: list[int],
    ) -> None:
        """
        Standard Q-learning (off-policy TD(0)) update:

            Q(s, a) ← Q(s, a) + α [ r + γ · max_a' Q(s', a') − Q(s, a) ]

        This converges to Q* under standard conditions (all state-action pairs
        visited infinitely often, decaying learning rate — we approximate the
        latter with epsilon decay reducing exploration gradually).
        """
        current_q = self._get_q(state, action)
        target = reward + self.hp.gamma * self._max_q(next_state, next_valid_actions)
        new_q = current_q + self.hp.alpha * (target - current_q)
        self._set_q(state, action, round(new_q, 6))

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(
        self,
        episodes: int = 5000,
        src_dst_pairs: Optional[list[tuple[int, int]]] = None,
        fluctuate_every: int = 50,
    ) -> list[EpisodeStats]:
        """
        Main training loop.

        Parameters
        ----------
        episodes : int
            Number of training episodes.
        src_dst_pairs : list[(src, dst)] | None
            If provided, randomly sample from these pairs each episode.
            If None, sample uniformly from all node pairs.
        fluctuate_every : int
            Every N episodes, randomly perturb congestion & loss rates
            to simulate a non-stationary environment.

        Returns
        -------
        list[EpisodeStats] — per-episode metrics for plotting.
        """
        nodes = list(self.network.graph.nodes())

        for ep in range(1, episodes + 1):
            # Pick source and destination
            if src_dst_pairs:
                src, dst = random.choice(src_dst_pairs)
            else:
                src, dst = random.sample(nodes, 2)

            # Run one episode
            stats = self._run_episode(src, dst, ep)
            self.training_history.append(stats)

            # Decay epsilon after each episode
            self._epsilon = max(self.hp.epsilon_min, self._epsilon * self.hp.epsilon_decay)

            # Periodically perturb the network to force adaptation
            if ep % fluctuate_every == 0:
                self.network.fluctuate_congestion(intensity=0.1)
                self.network.fluctuate_loss(intensity=0.01)

            # Log progress every 500 episodes
            if ep % 500 == 0:
                recent = self.training_history[-500:]
                avg_reward = sum(s.total_reward for s in recent) / len(recent)
                delivery_rate = sum(1 for s in recent if s.delivered) / len(recent)
                logger.info(
                    "Episode %5d/%d | avg_reward=%.2f | delivery=%.1f%% | ε=%.4f | Q-table=%d",
                    ep, episodes, avg_reward, delivery_rate * 100,
                    self._epsilon, len(self.q_table),
                )

        logger.info("Training complete. Q-table has %d entries.", len(self.q_table))
        return self.training_history

    def _run_episode(self, src: int, dst: int, episode_num: int) -> EpisodeStats:
        """Execute a single training episode from src to dst."""
        current = src
        state: State = (current, dst)
        visited: set[int] = {current}
        path: list[int] = [current]
        total_reward = 0.0
        delivered = False

        for step in range(self.hp.max_steps_per_episode):
            neighbors = self.network.get_neighbors(current)
            if not neighbors:
                break  # dead end

            # Select action (next hop)
            action = self.select_action(state, neighbors)

            # Compute reward
            reward = self.compute_reward(current, action, dst, visited)
            total_reward += reward

            # Observe next state
            next_node = action
            next_state: State = (next_node, dst)
            next_neighbors = self.network.get_neighbors(next_node)

            # Bellman update
            self.update(state, action, reward, next_state, next_neighbors)

            # Transition
            path.append(next_node)
            visited.add(next_node)
            current = next_node
            state = next_state

            # Check termination
            if current == dst:
                delivered = True
                break

        return EpisodeStats(
            episode=episode_num,
            total_reward=round(total_reward, 4),
            steps=len(path) - 1,
            path=path,
            delivered=delivered,
            epsilon=round(self._epsilon, 6),
        )

    # ------------------------------------------------------------------
    # Inference (post-training)
    # ------------------------------------------------------------------

    def route_packet(self, src: int, dst: int, max_steps: int = 50) -> RoutingResult:
        """
        Route a packet using the learned policy (pure exploitation, ε = 0).
        Returns a RoutingResult with the full path and cost breakdown.
        """
        current = src
        state: State = (current, dst)
        visited: set[int] = {current}
        path: list[int] = [current]
        total_cost = 0.0

        for _ in range(max_steps):
            neighbors = self.network.get_neighbors(current)
            if not neighbors:
                return RoutingResult(src, dst, path, total_cost, len(path) - 1,
                                     False, "no_neighbors")

            # Pure exploitation — pick best action from Q-table
            action = self.select_action(state, neighbors, exploit_only=True)

            # Accumulate actual edge cost (not reward — the real network metric)
            total_cost += self.network.get_edge_cost(current, action)

            path.append(action)
            if action == dst:
                return RoutingResult(src, dst, path, round(total_cost, 4),
                                     len(path) - 1, True, "delivered")

            if action in visited:
                return RoutingResult(src, dst, path, round(total_cost, 4),
                                     len(path) - 1, False, "loop")

            visited.add(action)
            current = action
            state = (current, dst)

        return RoutingResult(src, dst, path, round(total_cost, 4),
                             len(path) - 1, False, "max_steps")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_q_table_size(self) -> int:
        return len(self.q_table)

    def get_q_table_snapshot(self) -> dict:
        """Serialisable Q-table for API / frontend consumption."""
        entries = []
        for (state, action), q_val in sorted(self.q_table.items(), key=lambda x: -x[1])[:100]:
            entries.append({
                "current_node": state[0],
                "destination": state[1],
                "action": action,
                "q_value": round(q_val, 4),
            })
        return {"size": len(self.q_table), "top_entries": entries}
