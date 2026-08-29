# Code C · The Graph Layer (`graph/`)

Pipeline **step 2**: turn one window's flows into a `GraphState` — the snapshot
object the whole system passes around. Two files: `types.py` (the data shapes)
and `builder.py` (the construction).

---

## C.1 `graph/types.py` — NodeState, EdgeState, GraphState

### A node

```python
@dataclass
class NodeState:
    key: str            # stable id, e.g. "HOST-15" (or a hashed IP)
    label: str          # display name
    features: Vector    # the 8-number feature vector
    status: str = "normal"      # normal | deviating | anomalous
    is_server: bool = False
```

- One machine in one window. `features` is the vector from `window_node_features`.
- `status` starts "normal" and gets set later by the deviation engine.
- `is_server` flags "crown jewel" assets (used by the risk/propagation layers).

### An edge

```python
@dataclass
class EdgeState:
    src: str
    dst: str
    protocol: str
    features: Vector    # [packets, bytes, duration, packet_rate, byte_rate]
    weight: float = 1.0     # total packets — used as line thickness / traffic size
    dst_port: int = 0       # dominant destination port on this edge
```

- One directed connection `src → dst` in one window, with aggregated traffic
  features. `dst_port` is stored so the `BLOCK_PORT` counterfactual can target a
  specific service.

### The snapshot

```python
@dataclass
class GraphState:
    index: int
    window_start: float
    window_end: float
    nodes: dict[str, NodeState] = field(default_factory=dict)
    edges: list[EdgeState] = field(default_factory=list)
    node_feature_names: list[str] = field(default_factory=list)
    edge_feature_names: list[str] = field(default_factory=list)
```

- `field(default_factory=dict)` — gives each new `GraphState` its **own** empty
  dict/list. (You must never write `= {}` as a dataclass default; all instances
  would secretly share one object. `default_factory` creates a fresh one each
  time — an important Python gotcha.)
- `nodes` is keyed by node id for O(1) lookup; `edges` is a plain list.

Now the useful methods (helpers the rest of the code leans on):

```python
    def node_keys(self):      return sorted(self.nodes.keys())
    def node_count(self):     return len(self.nodes)
    def edge_count(self):     return len(self.edges)
```

- `node_keys()` returns node ids **sorted** — a deterministic order so feature
  matrices line up consistently every time.

```python
    def feature_matrix(self, keys=None):
        keys = keys if keys is not None else self.node_keys()
        return [list(self.nodes[k].features) if k in self.nodes else [0.0]*dim
                for k in keys]
```

- Stacks every node's feature vector into a **matrix** (list of rows), aligned to
  a fixed key order. This is exactly the `X` the model and normaliser consume.

```python
    def edge_set(self):   return {(e.src, e.dst) for e in self.edges}
    def adjacency(self):
        adj = {k: [] for k in self.nodes}
        for e in self.edges:
            adj.setdefault(e.src, []).append(e.dst)
        return adj
```

- `edge_set()` — the set of `(src, dst)` pairs. Used by the deviation engine's
  **Jaccard distance** (compare predicted vs actual edge sets).
- `adjacency()` — "who does each node point to." A dict `node → [neighbours]`.
  Used by propagation (spread along edges) and counterfactual (reachability).

```python
    def embedding(self):
        rows = self.feature_matrix()
        mean_feat = mean_vector(rows) if rows else []
        n = float(self.node_count()); e = float(self.edge_count())
        density = (e / (n * (n - 1))) if n > 1 else 0.0
        anomalous = sum(1 for nd in self.nodes.values() if nd.status == "anomalous")
        return list(mean_feat) + [n, e, density, float(anomalous)]
```

- The graph **embedding** (fingerprint) from the concepts part: the average node
  feature vector, plus four structural numbers — node count, edge count,
  **density** (how connected: actual edges ÷ maximum possible edges), and how many
  anomalous nodes. Because it's a *fixed length* regardless of how many nodes are
  present, two differently-sized graphs are still comparable — which is what the
  novelty engine needs.

```python
    def anomalous_keys(self): return [k for k,nd in self.nodes.items() if nd.status=="anomalous"]
    def server_keys(self):    return [k for k,nd in self.nodes.items() if nd.is_server]
    def clone(self):          ... deep copy of nodes+edges ...
    def to_json(self):        ... plain dict of everything ...
```

- `clone()` makes a **deep copy** so a counterfactual can modify a graph without
  touching the original (it rebuilds each NodeState/EdgeState with fresh lists).
- `to_json()` converts the graph to plain dicts/lists so it can be sent as an API
  response or stored in the database.

---

## C.2 `graph/builder.py` — flows → GraphState

```python
_SERVER_FANIN_THRESHOLD = 5

def _detect_servers(flows, nodes):
    fan_in = {n: set() for n in nodes}
    for f in flows:
        fan_in.setdefault(f.dst, set()).add(f.src)
    servers = set()
    for n in nodes:
        if n.upper().startswith("SERVER"):
            servers.add(n)
        elif len(fan_in.get(n, set())) >= _SERVER_FANIN_THRESHOLD:
            servers.add(n)
    return servers
```

- **Server detection.** A machine is treated as a server if its name starts with
  "SERVER" *or* if many *distinct* sources connect to it (**fan-in** ≥ 5). The
  intuition: servers are things lots of clients talk to. On hashed real data
  (where names are meaningless) the fan-in rule still finds the servers.
- `fan_in[n]` is a **set** of sources, so `len(...)` is the count of *distinct*
  clients.

```python
def build_graph_state(window, min_edge_weight=1.0):
    node_features = window_node_features(window.flows)
    nodes = sorted(node_features.keys())
    servers = _detect_servers(window.flows, nodes)
    graph = GraphState(index=window.index, window_start=window.start,
                       window_end=window.end,
                       node_feature_names=list(NODE_FEATURES),
                       edge_feature_names=list(EDGE_FEATURES))
    for n in nodes:
        graph.nodes[n] = NodeState(key=n, label=n, features=node_features[n],
                                   status="normal", is_server=(n in servers))
```

- Compute features (from `features.py`), detect servers, then create a
  `NodeState` for each node. Everything starts "normal."

```python
    agg = {}
    for f in window.flows:
        key = (f.src, f.dst)
        bucket = agg.setdefault(key, {"packets":0,"bytes":0,"duration":0.0,
                                      "protocols":Counter(),"ports":Counter()})
        bucket["packets"]  += f.packets
        bucket["bytes"]    += f.bytes
        bucket["duration"] += f.duration
        bucket["protocols"][f.protocol] += 1
        bucket["ports"][f.dst_port]     += 1
```

- **Edge aggregation.** Many flows can occur between the same pair of machines in
  one window; we collapse them into *one* edge carrying summed traffic. `agg` is
  keyed by the `(src, dst)` pair.
- `Counter` (from `collections`) is a dict that counts things. `bucket["ports"]
  [f.dst_port] += 1` tallies how often each port appears, so we can later pick the
  **most common** one.

```python
    for (src, dst), b in agg.items():
        weight = float(b["packets"])
        if weight < min_edge_weight:
            continue
        duration = b["duration"] if b["duration"] > 0 else 1e-6
        protocol = b["protocols"].most_common(1)[0][0]
        dst_port = b["ports"].most_common(1)[0][0]
        edge_feat = [float(b["packets"]), float(b["bytes"]), float(b["duration"]),
                     b["packets"]/duration, b["bytes"]/duration]
        graph.edges.append(EdgeState(src=src, dst=dst, protocol=protocol,
                                     features=edge_feat, weight=weight,
                                     dst_port=dst_port))
    return graph
```

- Turn each aggregated bucket into an `EdgeState`. `weight` = total packets;
  edges below `min_edge_weight` are dropped (removes trivial noise).
- `most_common(1)[0][0]` — `Counter.most_common(1)` returns `[(item, count)]`, so
  `[0][0]` is the most frequent item. We use it to pick the dominant protocol and
  port for the edge.
- `edge_feat` matches `EDGE_FEATURES` order: packets, bytes, duration, and the two
  per-second rates. `duration` is floored at `1e-6` to avoid divide-by-zero.

```python
def build_graph_sequence(flows, window_seconds, min_edge_weight=1.0):
    windows = build_windows(flows, window_seconds)
    return [build_graph_state(w, min_edge_weight=min_edge_weight) for w in windows]
```

- The top-level call: cut all flows into windows, build one `GraphState` per
  window, return the **sequence** `[G₁, G₂, ...]`. That list *is* the dynamic
  graph the world model learns from.

---

## Recap

Step 2 done: each time window is now a `GraphState` — a set of nodes with feature
vectors, a set of aggregated directed edges, and helper methods
(`feature_matrix`, `edge_set`, `adjacency`, `embedding`, `clone`) that the model
and analytics layers rely on. Servers are auto-detected. The whole run is a list
of these snapshots.

Next: [Code D — the world models](04-models.md)
