# Code F · The Analytics Engines (`analytics/`)

Pipeline **step 5**: the seven engines that turn a stream of deviation scores
into the "four questions" (spread, confidence, novelty, what-if) plus
interpretation (MITRE, explainability). Each is small and independent.

---

## F.1 `propagation.py` — the infection spreads

```python
@dataclass
class PropagationEvent:
    window_index: int
    source: str
    target: str
    propagation_velocity: float
    propagation_intensity: float
    effective_reproduction_number: float

def _infected(dev):
    return {k: d.deviation_score for k, d in dev.per_node.items()
            if d.status in ("anomalous", "deviating")}
```

- `_infected` = the set of "sick" nodes in a window (anomalous or deviating),
  with their scores. Treating deviating nodes as mildly infected lets us catch
  spread early.

```python
def compute_propagation(graphs, dev_by_index, window_seconds):
    events = []
    dt = window_seconds if window_seconds > 0 else 1.0
    by_index = {g.index: g for g in graphs}
    for t in sorted(dev_by_index):
        if (t - 1) not in dev_by_index: continue
        prev_inf = _infected(dev_by_index[t - 1])
        curr_inf = _infected(dev_by_index[t])
        newly = {k: v for k, v in curr_inf.items() if k not in prev_inf}
        if not newly or not prev_inf: continue

        velocity  = len(newly) / dt
        re        = len(newly) / max(len(prev_inf), 1)
        intensity = sum(newly.values()) / len(newly)

        adj = (by_index.get(t) or by_index.get(t - 1)).adjacency()
        for src in prev_inf:
            for tgt in adj.get(src, []):
                if tgt in newly:
                    events.append(PropagationEvent(t, src, tgt, velocity,
                                                   intensity, re))
    return events
```

Line by line — the epidemiology from concepts, in code:
- For each window `t` (that has a previous window), compare who was infected
  before (`prev_inf`) vs now (`curr_inf`). `newly` = the freshly infected.
- **velocity** = new infections per second (`len(newly) / dt`).
- **Rₑ** (effective reproduction number) = new infections divided by the number
  already infected. Rₑ > 1 → outbreak growing.
- **intensity** = average deviation score of the newly infected — how *strong*
  the spread is.
- **Attribution:** for each already-infected `src`, look at its neighbours in the
  graph (`adj[src]`); if a neighbour is newly infected, record a
  `src → tgt` event. That's how we reconstruct the path `HOST-15 → HOST-06 →
  HOST-07`. Only spreads *along an actual edge* count — an anomaly that appears
  with no infected neighbour isn't "propagation," it's independent.

---

## F.2 `uncertainty.py` — MC-Dropout confidence

```python
@dataclass
class UncertaintyResult:
    mean_prediction: float
    std_dev: float
    label: str            # LOW | MEDIUM | HIGH
    per_node_sigma: dict[str, float]

def estimate_uncertainty(model, history, num_passes=30, dropout=0.2, rng=None,
                         low_sigma=0.30, high_sigma=0.50):
    if num_passes < 2:
        raise ValueError("num_passes must be >= 2 to estimate variance")
    rng = rng or Rng(0)
    stacks = {}                                  # node → list of predicted vectors
    for p in range(num_passes):
        pred = model.predict_next(history, dropout=dropout, rng=rng.spawn(f"mc-{p}"))
        for key, node in pred.nodes.items():
            stacks.setdefault(key, []).append(list(node.features))

    per_node_sigma = {}
    node_means = []
    for key, rows in stacks.items():
        sigma_vec = std_vector(rows)             # spread across the passes
        per_node_sigma[key] = mean(sigma_vec)
        node_means.append(mean(mean_vector(rows)))

    graph_sigma = mean(per_node_sigma.values())
    return UncertaintyResult(mean(node_means), graph_sigma,
                             _label(graph_sigma, low_sigma, high_sigma),
                             per_node_sigma)
```

- This is **Monte-Carlo Dropout** from concepts. Run the model `num_passes` times,
  each with a *different random dropout* (via `rng.spawn(f"mc-{p}")` for
  reproducibility). Collect all predictions per node.
- `std_vector(rows)` gives the **spread** of a node's predictions across the
  passes. A big spread means the model is unsure. Average the spread across
  features (`per_node_sigma`) and across nodes (`graph_sigma`).
- `_label` turns the spread into LOW / MEDIUM / HIGH using the thresholds
  (calibrated to the z-scored space: <0.30 LOW, >0.50 HIGH). Because
  further-ahead forecasts vary more, uncertainty naturally grows with the horizon
  — exactly what an honest system should report.

---

## F.3 `novelty.py` — have we ever seen this? (OOD)

```python
_W_DIST = 0.5
_W_ERROR = 0.3
_W_UNCERTAINTY = 0.2
_SIGMA_SCALE = 0.1

class NoveltyScorer:
    def __init__(self, unusual=0.4, emerging=0.6, unknown=0.8): ...
    def fit(self, train_graphs):
        self._train_embeddings = [g.embedding() for g in train_graphs if g.node_count() > 0]
        # calibrate "how far is normal-far" = mean nearest-neighbour distance
        nn = []
        for i, e in enumerate(self._train_embeddings):
            best = min((euclidean(e, o) for j, o in enumerate(self._train_embeddings) if i != j),
                       default=None)
            if best is not None: nn.append(best)
        self._dist_scale = max(mean(nn) if nn else 1.0, 1e-6)
        self._fitted = True
        return self
```

- **Fit** stores the fingerprint (`embedding`) of every training graph and
  computes a **distance scale**: the average distance between a training graph and
  its *nearest* training neighbour. This is "how far apart normal graphs normally
  sit," so we can judge later distances *relative to normal variation* instead of
  an arbitrary constant.

```python
    def score(self, graph, prediction_error, uncertainty_sigma=0.0):
        emb = graph.embedding()
        dist = min(euclidean(emb, e) for e in self._train_embeddings)   # nearest normal
        novelty_dist = saturate(dist, self._dist_scale)
        err_term = clamp(prediction_error, 0, 1)
        unc_term = saturate(uncertainty_sigma, _SIGMA_SCALE)
        score = _W_DIST*novelty_dist + _W_ERROR*err_term + _W_UNCERTAINTY*unc_term
        return NoveltyResult(score, self._label(score), dist)
```

- **Score** blends three signals: how far this graph is from anything seen
  (distance, saturated by the calibrated scale), how badly the model forecast it
  (prediction error), and how unsure it was (uncertainty). The weighted sum is
  labelled on the scale **KNOWN → FAMILIAR → UNUSUAL → EMERGING → UNKNOWN** by
  `_label` (simple threshold bins). "UNKNOWN" is the system honestly flagging
  never-before-seen behaviour, even if it can't name the attack.

---

## F.4 `stability.py` — is the forecast robust?

```python
def _perturb(graph, sigma, rng):
    noisy = graph.clone()
    for node in noisy.nodes.values():
        node.features = [f + rng.gauss(0.0, sigma) for f in node.features]
    return noisy

def assess_stability(model, history, perturbation=0.03, num_trials=12,
                     unstable_threshold=0.12, rng=None):
    baseline = model.predict_next(history)
    deltas = []
    for t in range(num_trials):
        perturbed_last = _perturb(history[-1], perturbation, rng.spawn(f"stab-{t}"))
        pred = model.predict_next(list(history[:-1]) + [perturbed_last])
        deltas.append(_forecast_delta(baseline, pred))
    mean_delta = mean(deltas)
    return StabilityResult(stability_score=1.0/(1.0+mean_delta),
                           label="UNSTABLE" if mean_delta > unstable_threshold else "STABLE",
                           mean_delta=mean_delta)
```

- **Perturbation testing** from concepts. `_perturb` adds tiny Gaussian noise
  (`sigma = 0.03`) to the last window's features. We re-run the forecast
  `num_trials` times with different noise and measure how far each result drifts
  from the unperturbed `baseline` (`_forecast_delta` = average per-node Euclidean
  distance, normalised).
- `mean_delta` small → forecast barely moves → **STABLE**. Big → fragile →
  **UNSTABLE**. `stability_score = 1/(1+mean_delta)` maps this to (0, 1] where 1 =
  perfectly stable.

---

## F.5 `counterfactual.py` — the "what-if" engine

```python
VALID_ACTIONS = {"ISOLATE_NODE","BLOCK_EDGE","BLOCK_PORT","DISABLE_COMMUNICATION","RATE_LIMIT"}

@dataclass
class Intervention:
    action_type: str
    target_node: str | None = None
    target_edge: tuple[str, str] | None = None
    port: int | None = None
    rate_factor: float = 0.2

    def __post_init__(self):
        if self.action_type not in VALID_ACTIONS:
            raise ValueError(...)
        if self.action_type in ("ISOLATE_NODE","DISABLE_COMMUNICATION","RATE_LIMIT") and not self.target_node:
            raise ValueError(f"{self.action_type} requires 'target_node'")
        if self.action_type == "BLOCK_EDGE" and (not self.target_edge or len(self.target_edge) != 2):
            raise ValueError("BLOCK_EDGE requires 'target_edge' as [src, dst]")
        if self.action_type == "BLOCK_PORT" and self.port is None:
            raise ValueError("BLOCK_PORT requires 'port'")
        if self.action_type == "RATE_LIMIT" and not (0.0 <= self.rate_factor <= 1.0):
            raise ValueError("RATE_LIMIT 'rate_factor' must be in [0, 1]")
```

- `Intervention` describes a hypothetical defensive action. `X | None` means "an
  X or nothing" (an optional value).
- `__post_init__` runs automatically right after a dataclass is created. Here it
  **validates** the request: each action *must* have the parameter it operates on.
  This closes a real loophole — without it, `ISOLATE_NODE` with no target would
  silently do nothing and report "ΔRisk = 0," misleading the analyst into thinking
  isolation doesn't help. Now it raises a clear error (which the API turns into a
  400). **Fail loudly, never silently.**

```python
def apply_intervention(graph, iv):
    g = graph.clone()
    if iv.action_type == "ISOLATE_NODE":
        g.nodes.pop(iv.target_node, None)
        g.edges = [e for e in g.edges if e.src != iv.target_node and e.dst != iv.target_node]
    elif iv.action_type == "DISABLE_COMMUNICATION":
        g.edges = [e for e in g.edges if e.src != iv.target_node and e.dst != iv.target_node]
    elif iv.action_type == "BLOCK_EDGE":
        s, d = iv.target_edge
        g.edges = [e for e in g.edges if not (e.src == s and e.dst == d)]
    elif iv.action_type == "BLOCK_PORT":
        g.edges = [e for e in g.edges if not (e.dst_port == iv.port and (iv.target_node is None or e.dst == iv.target_node))]
    elif iv.action_type == "RATE_LIMIT":
        for e in g.edges:
            if e.src == iv.target_node:
                e.weight *= iv.rate_factor
                e.features = [f * iv.rate_factor for f in e.features]
    return g
```

- Each action edits a **cloned** graph (never the original). ISOLATE removes the
  node and all its edges; BLOCK_EDGE removes one connection; BLOCK_PORT removes
  connections on a port; RATE_LIMIT shrinks a node's outgoing traffic. This gives
  us the "modified world" Gₜ′.

```python
def _reachable(graph, sources):
    adj = graph.adjacency()
    seen = set()
    frontier = [s for s in sources if s in graph.nodes]
    while frontier:
        cur = frontier.pop()
        for nxt in adj.get(cur, []):
            if nxt not in seen:
                seen.add(nxt); frontier.append(nxt)
    return seen - set(sources)
```

- **Reachability by BFS/DFS** (breadth/depth-first search — following edges
  outward). Starting from the compromised nodes, which nodes can be reached by
  hopping along connections? This is the "how far could the attacker get" set.

```python
def compute_risk(graph, compromised):
    servers = set(graph.server_keys())
    reachable = _reachable(graph, compromised)
    server_risk  = len(reachable & servers) / len(servers) if servers else 0.0
    lateral_risk = len(reachable) / max(graph.node_count() - len(compromised), 1)
    # alternate_path_risk: fraction of servers reachable via >1 compromised entry
    ...
    overall = min(1.0, 0.6*server_risk + 0.4*lateral_risk)
    return {"db_risk": server_risk, "lateral_movement_risk": lateral_risk,
            "alternate_path_risk": alt, "overall": overall}
```

- **Risk** = how exposed the "crown jewel" servers and the wider estate are to the
  compromised nodes, via reachability. `db_risk` = fraction of servers reachable;
  `lateral_risk` = fraction of other machines reachable; `alternate_path_risk` =
  how resilient the attack is to blocking a single node. `overall` is a weighted
  blend in [0, 1].

```python
def run_counterfactual(model, history, intervention, compromised, horizon=3):
    forecast_a = model.predict_sequence(history, horizon)
    modified_last = apply_intervention(history[-1], intervention)
    forecast_b = model.predict_sequence(list(history[:-1]) + [modified_last], horizon)
    risk_a = compute_risk(forecast_a[-1], set(compromised))
    risk_b = compute_risk(forecast_b[-1], set(compromised))
    return CounterfactualResult(intervention.action_type, target, risk_a["overall"],
                                risk_b["overall"], risk_a["overall"] - risk_b["overall"],
                                risk_a, risk_b)
```

- The full "what-if" from concepts: forecast the future *as-is* (A), then forecast
  it *with the intervention applied to the current graph* (B), and compare the
  risk at the end of the horizon. **ΔRisk = Risk_A − Risk_B** is how much safer the
  action made the predicted future. Isolating the source early yields a big ΔRisk;
  doing it after the spread yields ~0 — the tool shows both honestly.

---

## F.6 `mitre.py` — naming the stage (interpretation only)

```python
ATTACK_STAGES = ["Reconnaissance","Initial Access","Execution","Persistence",
                 "Privilege Escalation","Discovery","Lateral Movement",
                 "Collection","Exfiltration","Command and Control"]

_FEATURE_TO_STAGE = {
    "failed_connections": "Discovery", "unique_ports": "Discovery",
    "unique_destinations": "Lateral Movement", "connection_frequency": "Lateral Movement",
    "mean_byte_rate": "Exfiltration", "outbound_ratio": "Exfiltration",
    "mean_packet_rate": "Command and Control", "mean_iat": "Command and Control",
}

def map_mitre_stage(top_features):
    candidates = [_FEATURE_TO_STAGE[f] for f in top_features[:2] if f in _FEATURE_TO_STAGE]
    if not candidates: return "Unknown"
    return max(candidates, key=lambda s: ATTACK_STAGES.index(s))
```

- A simple, honest **interpretation layer**: given the top contributing features
  (from `explain.py`), map them to a MITRE ATT&CK stage name. If two stages are
  implicated, it reports the **more advanced** one (`max` by position in
  `ATTACK_STAGES`) — an analyst cares most about how far along the attack is.
- Note: this makes *no* detection decision. It only labels what the maths already
  found, in familiar vocabulary.

---

## F.7 `explain.py` — which features drove the anomaly?

```python
EXPLANATION_CAVEAT = ("Feature contributions are indicative attributions of "
                      "forecast error, not a guaranteed-faithful causal explanation.")

def feature_contributions(predicted, actual, feature_names, top_k=None):
    sq = [(p - a) ** 2 for p, a in zip(predicted, actual)]
    total = sum(sq)
    if total <= 0:
        contributions = [{"feature": n, "contribution": 0.0} for n in feature_names]
    else:
        contributions = [{"feature": n, "contribution": s/total}
                         for n, s in zip(feature_names, sq)]
    contributions.sort(key=lambda c: c["contribution"], reverse=True)
    return contributions[:top_k] if top_k else contributions
```

- **Explainability:** attribute the anomaly to individual features by their
  *share of the squared error*. If `failed_connections` accounts for 70% of the
  prediction error, it gets contribution 0.70. Sorted highest-first, these tell
  the analyst *why* a node was flagged.
- `EXPLANATION_CAVEAT` is surfaced in the UI — an honesty note that these are
  *indicative* attributions, not proven cause. (The industrial upgrade is SHAP or
  attention analysis on a GNN.)
- `top_feature_names(...)` is a thin wrapper returning just the top-k names — fed
  into `map_mitre_stage` and the incident narration.

---

## Recap

Step 5 done. From the deviation timeline we now derive: **propagation** (spread +
Rₑ), **uncertainty** (MC-Dropout confidence), **novelty** (KNOWN→UNKNOWN),
**stability** (robustness), **counterfactuals** (what-if ΔRisk), plus
**MITRE** stage naming and per-feature **explanations**. Each engine is small,
independent, and honest about its limits.

Next: [Code G — persistence, narrative, pipeline, CLI](07-persistence.md)
