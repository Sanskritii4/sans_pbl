# AI-Driven Adaptive Packet Routing using Reinforcement Learning (Q-Learning)

## Technical Architecture Document

---

## 1. High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FRONTEND (React + Vite)                      │
│  ┌──────────┐ ┌──────────────┐ ┌────────────┐ ┌──────────────────┐ │
│  │ Topology │ │  Simulation  │ │ Metrics    │ │  Comparison      │ │
│  │ Viewer   │ │  Controls    │ │ Dashboard  │ │  Panel           │ │
│  └────┬─────┘ └──────┬───────┘ └─────┬──────┘ └────────┬─────────┘ │
│       │              │               │                  │           │
│       └──────────────┴───────────────┴──────────────────┘           │
│                              │  HTTP/REST + WebSocket               │
└──────────────────────────────┼──────────────────────────────────────┘
                               │
┌──────────────────────────────┼──────────────────────────────────────┐
│                       BACKEND (FastAPI + Python)                    │
│  ┌───────────────┐  ┌───────┴───────┐  ┌─────────────────────────┐ │
│  │  API Layer    │  │  WebSocket    │  │   Background Workers    │ │
│  │  (REST)       │  │  Manager      │  │   (asyncio tasks)       │ │
│  └───────┬───────┘  └───────┬───────┘  └────────────┬────────────┘ │
│          │                  │                        │              │
│  ┌───────┴──────────────────┴────────────────────────┴────────────┐ │
│  │                     SERVICE LAYER                              │ │
│  │  ┌──────────────┐ ┌──────────────┐ ┌────────────────────────┐ │ │
│  │  │  Network     │ │  RL Engine   │ │  Metrics Collector     │ │ │
│  │  │  Simulator   │ │  (Q-Agent)   │ │  & Comparator          │ │ │
│  │  └──────┬───────┘ └──────┬───────┘ └────────────┬───────────┘ │ │
│  └─────────┼────────────────┼──────────────────────┼─────────────┘ │
│            │                │                      │                │
│  ┌─────────┴────────────────┴──────────────────────┴─────────────┐ │
│  │                     CORE LAYER                                │ │
│  │  ┌─────────────┐ ┌────────────────┐ ┌───────────────────────┐ │ │
│  │  │  NetworkX   │ │  Q-Table       │ │  Dijkstra Baseline    │ │ │
│  │  │  Graph Mgr  │ │  Store         │ │  Engine               │ │ │
│  │  └─────────────┘ └────────────────┘ └───────────────────────┘ │ │
│  └───────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────┘
```

### Architecture Principles

- **Separation of Concerns**: API layer → Service layer → Core layer. No business logic in endpoints.
- **Stateful Simulation**: The network graph, link weights, and Q-table persist in-memory across API calls within a session.
- **Real-time Push**: WebSocket channel pushes live metrics and topology changes to the frontend without polling.
- **Comparability**: Every routing decision by the RL agent produces a parallel Dijkstra result for direct comparison.

---

## 2. Backend Architecture (Module-Level Design)

### Module Breakdown

| Module | Responsibility |
|---|---|
| `api/` | FastAPI routers, request/response schemas, WebSocket endpoints |
| `core/network.py` | NetworkX graph creation, manipulation, dynamic weight updates |
| `core/dijkstra.py` | Static shortest-path routing using Dijkstra's algorithm |
| `core/q_agent.py` | Q-learning agent: Q-table, ε-greedy policy, training loop |
| `core/environment.py` | MDP environment wrapping the network graph |
| `services/simulator.py` | Orchestrates packet simulation (packet generation, step-by-step routing) |
| `services/metrics.py` | Collects, aggregates, and stores performance metrics |
| `services/comparator.py` | Runs parallel Dijkstra vs Q-learning routing and computes deltas |
| `models/` | Pydantic schemas for all data structures |
| `config.py` | Centralized configuration (hyperparameters, network topology presets) |
| `main.py` | FastAPI app factory, middleware, lifespan events |

### Core Module Details

#### `core/network.py`

```python
class NetworkManager:
    """Manages the NetworkX graph representing the network topology."""

    def __init__(self, topology: str = "default"):
        self.graph: nx.DiGraph = nx.DiGraph()
        self._build_topology(topology)

    def _build_topology(self, preset: str) -> None:
        """Build from preset: 'grid_4x4', 'mesh_6', 'random_10', 'custom'."""

    def get_neighbors(self, node: int) -> list[int]:
        """Return adjacent nodes for a given node."""

    def get_edge_weight(self, src: int, dst: int) -> float:
        """Return current weight (latency/cost) of an edge."""

    def update_edge_weight(self, src: int, dst: int, weight: float) -> None:
        """Dynamically update an edge weight (simulates congestion)."""

    def inject_failure(self, src: int, dst: int) -> None:
        """Remove an edge to simulate link failure."""

    def restore_link(self, src: int, dst: int, weight: float) -> None:
        """Restore a previously failed link."""

    def get_topology_data(self) -> dict:
        """Return serializable {nodes: [...], edges: [...]} for frontend."""
```

#### `core/dijkstra.py`

```python
class DijkstraRouter:
    """Static shortest-path routing baseline."""

    def __init__(self, network: NetworkManager):
        self.network = network

    def find_shortest_path(self, src: int, dst: int) -> tuple[list[int], float]:
        """Returns (path, total_cost) using nx.dijkstra_path."""

    def route_packet(self, src: int, dst: int) -> RoutingResult:
        """Route a single packet and return full result with metrics."""
```

#### `core/q_agent.py` — *Detailed in Section 3*

#### `core/environment.py`

```python
class NetworkEnvironment:
    """MDP Environment for the Q-learning agent."""

    def __init__(self, network: NetworkManager, destination: int):
        self.network = network
        self.destination = destination
        self.current_node: int = -1

    def reset(self, start_node: int) -> State:
        """Reset environment, return initial state."""

    def step(self, action: int) -> tuple[State, float, bool]:
        """Execute action, return (next_state, reward, done)."""

    def get_valid_actions(self, state: State) -> list[int]:
        """Return valid next-hop nodes from current state."""
```

---

## 3. RL Agent Design — MDP Formulation

### Markov Decision Process (MDP)

```
MDP = (S, A, P, R, γ)
```

#### State Space (S)

```
S = (current_node, destination_node, congestion_vector)
```

| Component | Type | Description |
|---|---|---|
| `current_node` | `int` | ID of the node where the packet currently resides |
| `destination_node` | `int` | ID of the target node |
| `congestion_vector` | `tuple[int, ...]` | Discretized congestion levels (0=low, 1=med, 2=high) for each link from current_node |

**State encoding** (for Q-table indexing):

```python
state_key = (current_node, destination_node, tuple(congestion_levels))
```

- For a 10-node network: |S| ≈ 10 × 10 × 3^max_degree ≈ manageable for tabular Q-learning.

#### Action Space (A)

```
A(s) = {neighbor_node | neighbor_node ∈ adj(current_node)}
```

- Variable-size action space per state: only valid next-hop neighbors.
- Invalid actions are masked (not presented to the agent).

#### Transition Function (P)

```
P(s' | s, a) = deterministic in base case
             = stochastic when congestion_vector changes between steps
```

- **Deterministic component**: `a` moves the packet to `next_node = a`.
- **Stochastic component**: Congestion levels may change between steps (simulated by the environment).

#### Reward Function (R)

```python
def compute_reward(self, current: int, next_hop: int, destination: int, 
                   visited: set) -> float:
    edge_cost = self.network.get_edge_weight(current, next_hop)

    if next_hop == destination:
        return +100.0                         # Goal reached

    if next_hop in visited:
        return -50.0                          # Loop penalty

    hop_penalty = -1.0                        # Per-hop cost
    latency_penalty = -edge_cost              # Proportional to link weight
    
    # Progress reward: closer to destination → small bonus
    dist_before = nx.shortest_path_length(G, current, destination, weight='weight')
    dist_after  = nx.shortest_path_length(G, next_hop, destination, weight='weight')
    progress = (dist_before - dist_after) * 2.0

    return hop_penalty + latency_penalty + progress
```

| Reward Component | Value | Purpose |
|---|---|---|
| Goal reached | `+100.0` | Strong incentive to reach destination |
| Loop detected | `-50.0` | Prevent cyclic routing |
| Per-hop cost | `-1.0` | Minimize path length |
| Latency penalty | `-edge_weight` | Prefer low-latency links |
| Progress bonus | `+2 × Δdistance` | Guide agent toward destination |

#### Policy (π)

**ε-greedy with decay:**

```python
def select_action(self, state: State, valid_actions: list[int]) -> int:
    if random.random() < self.epsilon:
        return random.choice(valid_actions)           # Explore
    
    q_values = {a: self.q_table.get((state, a), 0.0) for a in valid_actions}
    return max(q_values, key=q_values.get)             # Exploit
```

#### Q-Table Update Rule

```
Q(s, a) ← Q(s, a) + α [r + γ · max_a' Q(s', a') - Q(s, a)]
```

| Hyperparameter | Symbol | Default | Range |
|---|---|---|---|
| Learning rate | α | 0.1 | [0.01, 0.5] |
| Discount factor | γ | 0.95 | [0.8, 0.99] |
| Epsilon (initial) | ε₀ | 1.0 | — |
| Epsilon (min) | ε_min | 0.01 | — |
| Epsilon decay | δ | 0.995 | [0.99, 0.999] |
| Max steps per episode | — | 50 | — |
| Training episodes | — | 5000 | [1000, 20000] |

### Q-Agent Implementation

```python
class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.95, epsilon=1.0,
                 epsilon_min=0.01, epsilon_decay=0.995):
        self.q_table: dict[tuple[State, int], float] = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.training_history: list[EpisodeMetrics] = []

    def train(self, env: NetworkEnvironment, episodes: int = 5000) -> TrainingResult:
        """Full training loop. Returns convergence metrics."""

    def select_action(self, state: State, valid_actions: list[int]) -> int:
        """ε-greedy action selection."""

    def update(self, state: State, action: int, reward: float,
               next_state: State, next_valid_actions: list[int]) -> None:
        """Single Q-value update step."""

    def route_packet(self, src: int, dst: int, env: NetworkEnvironment) -> RoutingResult:
        """Route using learned policy (greedy, ε=0)."""

    def get_q_table_snapshot(self) -> dict:
        """Serializable Q-table for visualization."""

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
```

---

## 4. API Design

**Base URL**: `http://localhost:8000/api/v1`

### 4.1 Network Topology

#### `GET /network/topology`
> Retrieve the current network graph.

**Response** `200`:
```json
{
  "nodes": [
    {"id": 0, "label": "Router-0", "x": 100, "y": 200, "congestion": 0.3},
    {"id": 1, "label": "Router-1", "x": 250, "y": 100, "congestion": 0.7}
  ],
  "edges": [
    {"source": 0, "target": 1, "weight": 3.5, "congestion_level": "low"},
    {"source": 1, "target": 2, "weight": 7.2, "congestion_level": "high"}
  ],
  "metadata": {
    "node_count": 10,
    "edge_count": 18,
    "topology_type": "mesh_6"
  }
}
```

#### `POST /network/topology`
> Create/reset the network with a given topology preset.

**Request**:
```json
{
  "topology_type": "grid_4x4",
  "custom_edges": null
}
```

**Response** `201`: Same as GET response.

#### `PUT /network/edge`
> Dynamically update an edge weight (simulate congestion).

**Request**:
```json
{
  "source": 0,
  "target": 1,
  "weight": 12.5
}
```

**Response** `200`:
```json
{
  "source": 0,
  "target": 1,
  "old_weight": 3.5,
  "new_weight": 12.5,
  "timestamp": "2026-02-16T21:00:00Z"
}
```

#### `POST /network/failure`
> Inject a link failure.

**Request**:
```json
{
  "source": 2,
  "target": 5,
  "action": "fail"
}
```

**Response** `200`:
```json
{
  "source": 2,
  "target": 5,
  "status": "removed",
  "remaining_edges": 17
}
```

---

### 4.2 RL Agent / Training

#### `POST /agent/train`
> Start training the Q-learning agent.

**Request**:
```json
{
  "episodes": 5000,
  "alpha": 0.1,
  "gamma": 0.95,
  "epsilon": 1.0,
  "epsilon_min": 0.01,
  "epsilon_decay": 0.995,
  "source_node": null,
  "destination_node": null
}
```

**Response** `202`:
```json
{
  "task_id": "train_abc123",
  "status": "training_started",
  "config": {
    "episodes": 5000,
    "alpha": 0.1,
    "gamma": 0.95
  }
}
```

#### `GET /agent/training-status/{task_id}`
> Poll training progress.

**Response** `200`:
```json
{
  "task_id": "train_abc123",
  "status": "in_progress",
  "progress": {
    "current_episode": 2340,
    "total_episodes": 5000,
    "percent_complete": 46.8
  },
  "live_metrics": {
    "avg_reward_last_100": 67.3,
    "epsilon_current": 0.42,
    "convergence_delta": 0.003
  }
}
```

#### `GET /agent/q-table`
> Retrieve the current Q-table (or a summary of it).

**Response** `200`:
```json
{
  "q_table_size": 1840,
  "sample_entries": [
    {
      "state": {"current_node": 0, "destination": 9, "congestion": [0, 1, 0]},
      "action": 1,
      "q_value": 73.21
    }
  ],
  "heatmap_data": {
    "nodes": [0, 1, 2, 3],
    "destinations": [9],
    "max_q_per_node": [73.2, 68.1, 81.4, 55.0]
  }
}
```

#### `GET /agent/hyperparameters`
> Retrieve current hyperparameter configuration.

**Response** `200`:
```json
{
  "alpha": 0.1,
  "gamma": 0.95,
  "epsilon": 0.03,
  "epsilon_min": 0.01,
  "epsilon_decay": 0.995,
  "max_steps_per_episode": 50,
  "training_episodes_completed": 5000
}
```

---

### 4.3 Routing

#### `POST /route/q-learning`
> Route a packet using the trained Q-learning policy.

**Request**:
```json
{
  "source": 0,
  "destination": 9,
  "packet_id": "pkt_001"
}
```

**Response** `200`:
```json
{
  "packet_id": "pkt_001",
  "algorithm": "q-learning",
  "path": [0, 3, 6, 8, 9],
  "total_cost": 14.7,
  "hop_count": 4,
  "latency_ms": 14.7,
  "per_hop_details": [
    {"from": 0, "to": 3, "edge_cost": 3.2, "q_value": 73.2},
    {"from": 3, "to": 6, "edge_cost": 4.1, "q_value": 68.5},
    {"from": 6, "to": 8, "edge_cost": 3.9, "q_value": 52.1},
    {"from": 8, "to": 9, "edge_cost": 3.5, "q_value": 100.0}
  ]
}
```

#### `POST /route/dijkstra`
> Route a packet using Dijkstra's algorithm.

**Request**:
```json
{
  "source": 0,
  "destination": 9,
  "packet_id": "pkt_001"
}
```

**Response** `200`:
```json
{
  "packet_id": "pkt_001",
  "algorithm": "dijkstra",
  "path": [0, 1, 4, 7, 9],
  "total_cost": 13.2,
  "hop_count": 4,
  "latency_ms": 13.2
}
```

#### `POST /route/compare`
> Route a packet with BOTH algorithms and return comparison.

**Request**:
```json
{
  "source": 0,
  "destination": 9,
  "num_packets": 50,
  "congestion_variation": true
}
```

**Response** `200`:
```json
{
  "comparison_id": "cmp_xyz789",
  "num_packets": 50,
  "results": {
    "q_learning": {
      "avg_cost": 15.3,
      "avg_hops": 4.2,
      "avg_latency_ms": 15.3,
      "packets_delivered": 50,
      "packets_dropped": 0
    },
    "dijkstra": {
      "avg_cost": 14.1,
      "avg_hops": 3.8,
      "avg_latency_ms": 14.1,
      "packets_delivered": 48,
      "packets_dropped": 2
    },
    "delta": {
      "cost_diff_percent": 8.5,
      "hop_diff_percent": 10.5,
      "delivery_rate_diff": -4.0,
      "winner_by_delivery": "q-learning",
      "winner_by_cost": "dijkstra"
    }
  }
}
```

---

### 4.4 Simulation

#### `POST /simulation/start`
> Start a continuous simulation with dynamic congestion.

**Request**:
```json
{
  "packets_per_second": 5,
  "duration_seconds": 60,
  "congestion_model": "random_walk",
  "failure_probability": 0.05
}
```

**Response** `202`:
```json
{
  "simulation_id": "sim_001",
  "status": "running",
  "config": {
    "packets_per_second": 5,
    "duration_seconds": 60
  }
}
```

#### `GET /simulation/status/{simulation_id}`

**Response** `200`:
```json
{
  "simulation_id": "sim_001",
  "status": "running",
  "elapsed_seconds": 23,
  "packets_routed": {"q_learning": 115, "dijkstra": 115},
  "live_metrics": {
    "current_avg_latency_ql": 12.3,
    "current_avg_latency_dj": 11.8
  }
}
```

#### `POST /simulation/stop/{simulation_id}`

**Response** `200`:
```json
{
  "simulation_id": "sim_001",
  "status": "stopped",
  "final_report": { "...see metrics section..." }
}
```

---

### 4.5 Metrics

#### `GET /metrics/history`
> Retrieve historical metrics for charting.

**Query Params**: `?window=last_100&metric=avg_cost`

**Response** `200`:
```json
{
  "metric": "avg_cost",
  "data_points": [
    {"episode": 100, "q_learning": 45.2, "dijkstra": 13.1, "timestamp": "..."},
    {"episode": 200, "q_learning": 32.1, "dijkstra": 13.1, "timestamp": "..."},
    {"episode": 300, "q_learning": 18.7, "dijkstra": 13.2, "timestamp": "..."}
  ]
}
```

---

### 4.6 WebSocket

#### `WS /ws/live`
> Real-time event stream for frontend.

**Server → Client message types:**

```json
{"type": "topology_update", "data": {"edge": [2,5], "new_weight": 8.3}}
{"type": "packet_routed", "data": {"packet_id": "pkt_042", "algorithm": "q-learning", "path": [0,3,6,9], "cost": 12.1}}
{"type": "training_progress", "data": {"episode": 1500, "avg_reward": 54.2, "epsilon": 0.31}}
{"type": "link_failure", "data": {"source": 2, "target": 5}}
{"type": "metrics_update", "data": {"q_avg_latency": 13.2, "dj_avg_latency": 11.9}}
```

---

## 5. Frontend Architecture (Component Hierarchy)

```
<App>
├── <Header />                        # App title, nav, status indicator
├── <MainLayout>
│   ├── <Sidebar>
│   │   ├── <TopologySelector />      # Preset topology chooser
│   │   ├── <HyperparameterPanel />   # α, γ, ε sliders
│   │   ├── <TrainingControls />      # Train, Stop, Reset buttons
│   │   └── <SimulationControls />    # Start/stop sim, packet rate slider
│   │
│   ├── <ContentArea>
│   │   ├── <TabBar />                # Topology | Training | Routing | Comparison
│   │   │
│   │   ├── <TopologyView>            # Tab 1
│   │   │   ├── <NetworkGraph />      # vis.js/d3 force-directed graph
│   │   │   ├── <NodeInfoPanel />     # Selected node details
│   │   │   └── <EdgeEditor />        # Click-to-edit edge weights
│   │   │
│   │   ├── <TrainingView>            # Tab 2
│   │   │   ├── <RewardChart />       # Reward vs episode (Recharts)
│   │   │   ├── <EpsilonChart />      # ε-decay over time
│   │   │   ├── <ConvergenceChart />  # Q-value convergence
│   │   │   └── <QTableHeatmap />     # Q-value heatmap per node
│   │   │
│   │   ├── <RoutingView>             # Tab 3
│   │   │   ├── <PacketForm />        # Source/dest selector, send button
│   │   │   ├── <PathAnimator />      # Animated packet traversal on graph
│   │   │   └── <RouteDetails />      # Per-hop breakdown table
│   │   │
│   │   └── <ComparisonView>          # Tab 4
│   │       ├── <MetricsBarChart />   # Side-by-side bar chart
│   │       ├── <MetricsTable />      # Detailed comparison table
│   │       └── <WinnerBadge />       # Which algorithm won & why
│   │
│   └── <StatusBar />                 # Connection status, sim clock, packet count
│
└── <Footer />
```

### Key Frontend Libraries

| Library | Purpose |
|---|---|
| `react` + `react-dom` | Core UI |
| `vite` | Build tooling |
| `react-router-dom` | Client-side routing |
| `vis-network` or `react-force-graph` | Network graph visualization |
| `recharts` | Training/metric charts |
| `zustand` | Lightweight global state management |
| `axios` | HTTP client |
| `react-hot-toast` | Notifications |

---

## 6. Data Flow

```
┌───────────┐       ┌───────────────────┐       ┌──────────────┐
│  Frontend │       │     FastAPI        │       │   Core       │
│  (React)  │       │     Backend        │       │   Engine     │
└─────┬─────┘       └────────┬──────────┘       └──────┬───────┘
      │                      │                         │
      │  1. POST /network    │                         │
      │     /topology        │                         │
      │─────────────────────>│  2. NetworkManager      │
      │                      │     .build_topology()   │
      │                      │────────────────────────>│
      │                      │  3. return graph data   │
      │                      │<────────────────────────│
      │  4. topology JSON    │                         │
      │<─────────────────────│                         │
      │                      │                         │
      │  5. POST /agent      │                         │
      │     /train           │                         │
      │─────────────────────>│  6. spawn async task    │
      │                      │────────────────────────>│
      │  7. {task_id}        │     QLearningAgent      │
      │<─────────────────────│     .train()            │
      │                      │                         │
      │  8. WS /ws/live      │  9. training_progress   │
      │<═════════════════════│     events pushed       │
      │  (real-time updates) │<────────────────────────│
      │                      │                         │
      │  10. POST /route     │                         │
      │     /compare         │                         │
      │─────────────────────>│  11. Q-Agent.route()    │
      │                      │  12. Dijkstra.route()   │
      │                      │────────────────────────>│
      │                      │  13. both results       │
      │                      │<────────────────────────│
      │  14. comparison JSON │                         │
      │<─────────────────────│                         │
      │                      │                         │
```

### State Management Flow (Frontend)

```
User Input → Component Event Handler
    → Zustand Action (API call)
        → Zustand Store Update
            → React Re-render (subscribed components)

WebSocket Message → WS Handler
    → Zustand Store Update (live metrics/topology)
        → React Re-render (charts, graph)
```

---

## 7. Performance Metrics Tracking Design

### Metrics Schema

```python
@dataclass
class RoutingMetrics:
    packet_id: str
    algorithm: str              # "q-learning" | "dijkstra"
    source: int
    destination: int
    path: list[int]
    hop_count: int
    total_cost: float           # Sum of edge weights along path
    latency_ms: float           # Simulated end-to-end latency
    delivered: bool             # Did packet reach destination?
    timestamp: datetime

@dataclass
class TrainingMetrics:
    episode: int
    total_reward: float
    steps_taken: int
    epsilon: float
    avg_q_value: float          # Mean of all Q-values in table
    q_value_delta: float        # Max change in Q-values this episode
    path_found: bool
    path_length: int

@dataclass
class ComparisonSummary:
    num_packets: int
    q_learning_avg_cost: float
    dijkstra_avg_cost: float
    q_learning_avg_hops: float
    dijkstra_avg_hops: float
    q_learning_delivery_rate: float  # packets_delivered / total
    dijkstra_delivery_rate: float
    q_learning_adaptability_score: float  # performance under dynamic congestion
    dijkstra_adaptability_score: float
```

### Metrics Tracked Per Experiment

| Metric | Q-Learning | Dijkstra | Purpose |
|---|---|---|---|
| Average path cost | ✓ | ✓ | Routing efficiency |
| Average hop count | ✓ | ✓ | Path optimality |
| Packet delivery ratio | ✓ | ✓ | Reliability |
| End-to-end latency | ✓ | ✓ | Performance |
| Throughput (pkts/sec) | ✓ | ✓ | Capacity |
| Adaptability to failures | ✓ | ✓ | Resilience (key RL advantage) |
| Convergence episode | ✓ | — | Training efficiency |
| Q-value variance | ✓ | — | Stability |
| Re-training speed | ✓ | — | Adaptation speed |

### Storage Strategy

- **In-memory**: `collections.deque(maxlen=10000)` for live metrics during simulation.
- **File-based**: Periodic dump to JSON/CSV in `data/experiments/` for post-analysis.
- **No external DB required** for a university project; SQLite optional for persistence.

---

## 8. Folder Structure

```
sans_pbl/
├── backend/
│   ├── main.py                        # FastAPI app entry point
│   ├── config.py                      # Settings, hyperparameter defaults
│   ├── requirements.txt               # Python dependencies
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── router_network.py          # /network/* endpoints
│   │   ├── router_agent.py            # /agent/* endpoints
│   │   ├── router_routing.py          # /route/* endpoints
│   │   ├── router_simulation.py       # /simulation/* endpoints
│   │   ├── router_metrics.py          # /metrics/* endpoints
│   │   └── websocket.py               # WebSocket handler
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── network.py                 # NetworkX graph manager
│   │   ├── dijkstra.py                # Dijkstra routing engine
│   │   ├── q_agent.py                 # Q-learning agent
│   │   └── environment.py             # MDP environment
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── simulator.py               # Packet simulation orchestrator
│   │   ├── metrics.py                 # Metrics collection & aggregation
│   │   └── comparator.py              # Algorithm comparison engine
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── network_models.py          # Pydantic: topology schemas
│   │   ├── agent_models.py            # Pydantic: training config, Q-table
│   │   ├── routing_models.py          # Pydantic: routing request/response
│   │   └── metrics_models.py          # Pydantic: metrics schemas
│   │
│   ├── data/
│   │   ├── topologies/                # Preset topology JSON files
│   │   │   ├── grid_4x4.json
│   │   │   ├── mesh_6.json
│   │   │   └── random_10.json
│   │   └── experiments/               # Saved experiment results
│   │       └── .gitkeep
│   │
│   └── tests/
│       ├── test_network.py
│       ├── test_q_agent.py
│       ├── test_dijkstra.py
│       ├── test_api.py
│       └── test_comparator.py
│
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   │
│   ├── public/
│   │   └── favicon.ico
│   │
│   └── src/
│       ├── main.jsx                   # React entry point
│       ├── App.jsx                    # Root component + routing
│       ├── index.css                  # Global styles
│       │
│       ├── components/
│       │   ├── Header.jsx
│       │   ├── Sidebar/
│       │   │   ├── TopologySelector.jsx
│       │   │   ├── HyperparameterPanel.jsx
│       │   │   ├── TrainingControls.jsx
│       │   │   └── SimulationControls.jsx
│       │   ├── Topology/
│       │   │   ├── NetworkGraph.jsx
│       │   │   ├── NodeInfoPanel.jsx
│       │   │   └── EdgeEditor.jsx
│       │   ├── Training/
│       │   │   ├── RewardChart.jsx
│       │   │   ├── EpsilonChart.jsx
│       │   │   ├── ConvergenceChart.jsx
│       │   │   └── QTableHeatmap.jsx
│       │   ├── Routing/
│       │   │   ├── PacketForm.jsx
│       │   │   ├── PathAnimator.jsx
│       │   │   └── RouteDetails.jsx
│       │   └── Comparison/
│       │       ├── MetricsBarChart.jsx
│       │       ├── MetricsTable.jsx
│       │       └── WinnerBadge.jsx
│       │
│       ├── stores/
│       │   ├── networkStore.js        # Zustand: topology state
│       │   ├── agentStore.js          # Zustand: training state
│       │   ├── routingStore.js        # Zustand: routing results
│       │   └── metricsStore.js        # Zustand: live metrics
│       │
│       ├── services/
│       │   ├── api.js                 # Axios HTTP client wrapper
│       │   └── websocket.js           # WebSocket connection manager
│       │
│       └── utils/
│           ├── constants.js
│           └── formatters.js
│
├── docs/
│   ├── ARCHITECTURE.md                # This document
│   └── API_REFERENCE.md               # Auto-generated or manual
│
├── scripts/
│   ├── start_dev.sh                   # Start both frontend + backend
│   └── seed_topology.py               # Generate preset topologies
│
├── docker-compose.yml                 # Local multi-container setup
├── Dockerfile.backend
├── Dockerfile.frontend
├── .gitignore
└── README.md
```

---

## 9. Deployment Strategy

### 9.1 Local Development

```bash
# Terminal 1 — Backend
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 — Frontend
cd frontend
npm install
npm run dev    # Vite dev server on port 5173
```

**Proxy config** (`vite.config.js`):
```js
export default defineConfig({
  server: {
    proxy: {
      '/api': 'http://localhost:8000',
      '/ws': { target: 'ws://localhost:8000', ws: true }
    }
  }
})
```

### 9.2 Docker (Local Production-Like)

```yaml
# docker-compose.yml
version: "3.9"
services:
  backend:
    build:
      context: .
      dockerfile: Dockerfile.backend
    ports:
      - "8000:8000"
    volumes:
      - ./backend/data:/app/data
    environment:
      - ENV=production

  frontend:
    build:
      context: .
      dockerfile: Dockerfile.frontend
    ports:
      - "3000:80"    # Nginx serves built React app
    depends_on:
      - backend
```

**Dockerfile.backend**:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/ .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Dockerfile.frontend**:
```dockerfile
FROM node:20-alpine AS build
WORKDIR /app
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
```

### 9.3 Cloud Deployment Options

| Platform | Backend | Frontend | Pros |
|---|---|---|---|
| **Railway** | Python service | Static site | Zero-config, free tier, git deploy |
| **Render** | Web service | Static site | Auto-deploy from GitHub, free tier |
| **AWS (EC2 + S3/CloudFront)** | EC2 t2.micro | S3 static hosting | Full control, free tier eligible |
| **Google Cloud Run** | Container | Firebase Hosting | Auto-scaling, serverless |

### Recommended Cloud Setup (Railway — simplest for university projects)

```
GitHub Repo
    ├── Push to main
    │
    ├──> Railway Backend Service
    │      - Auto-detect Python
    │      - Start cmd: uvicorn main:app --host 0.0.0.0 --port $PORT
    │      - Environment vars: ENV=production
    │
    └──> Railway Static Site (or Vercel/Netlify for frontend)
           - Build cmd: npm run build
           - Output dir: dist
           - Env: VITE_API_URL=https://backend-url.railway.app
```

### Environment Variables

| Variable | Dev | Production |
|---|---|---|
| `BACKEND_HOST` | `localhost` | `0.0.0.0` |
| `BACKEND_PORT` | `8000` | `$PORT` |
| `CORS_ORIGINS` | `["http://localhost:5173"]` | `["https://your-domain.com"]` |
| `ENV` | `development` | `production` |
| `VITE_API_URL` | `/api` (proxied) | `https://api.your-domain.com` |

---

## Appendix: Dependency List

### Backend (`requirements.txt`)

```
fastapi==0.109.0
uvicorn[standard]==0.27.0
networkx==3.2.1
numpy==1.26.3
pydantic==2.5.3
websockets==12.0
python-dotenv==1.0.0
pytest==7.4.4
httpx==0.26.0
```

### Frontend (`package.json` dependencies)

```json
{
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-router-dom": "^6.21.0",
    "recharts": "^2.10.0",
    "vis-network": "^9.1.6",
    "vis-data": "^7.1.9",
    "zustand": "^4.4.7",
    "axios": "^1.6.5",
    "react-hot-toast": "^2.4.1"
  },
  "devDependencies": {
    "@vitejs/plugin-react": "^4.2.1",
    "vite": "^5.0.0"
  }
}
```
