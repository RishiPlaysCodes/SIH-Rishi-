# Code E · Forecasting & the Deviation Score (`forecast/`)

Pipeline **step 4** — the moment forecasting becomes *detection*. Two files:
`deviation.py` (measure the gap between prediction and reality) and `engine.py`
(drive the model over the whole sequence).

Recall the core equation: **Dₜ = d(Gₜ, Ĝₜ)** — the distance between the *real*
graph and the *predicted* graph. A big distance = "the network did something we
didn't see coming" = suspicious.

---

## E.1 `forecast/deviation.py`

### The tuning constants

```python
DEFAULT_WEIGHTS = {
    "feature": 0.5, "node_state": 0.18, "temporal": 0.15,
    "structural": 0.10, "edge_state": 0.07,
}

_RMS_WEIGHT = 0.4
_MAX_WEIGHT = 0.6

_SAT_FEATURE = 2.5
_SAT_NODE = 3.5
_SAT_TEMPORAL = 3.0
```

- `DEFAULT_WEIGHTS` blends the five error components into one score. **feature**
  error dominates (0.5) because the forecast error is the core signal; the others
  add context. These weights were *calibrated* against known attacks to reach
  ~0.98 precision.
- `_RMS_WEIGHT`/`_MAX_WEIGHT` blend "average error" with "worst-feature error"
  (explained below).
- The `_SAT_*` constants set "how many standard deviations of error counts as
  fully anomalous" for each component (used by `saturate`).

### The saturating function

```python
def saturate(x, scale):
    x = abs(x)
    return x / (x + scale) if (x + scale) > 0 else 0.0
```

- The bounded squashing from concepts: maps any non-negative error into **[0, 1)**.
- Example with `scale = 2.5`: an error of 2.5 → 0.5; error 10 → 0.8; error 100 →
  0.976. It creeps toward 1 but never exceeds it. This is what keeps z-score-based
  (unbounded) errors readable as a 0–1 score.

### Feature error: average blended with the worst dimension

```python
def _rms(residuals):
    return math.sqrt(sum(r*r for r in residuals) / len(residuals)) if residuals else 0.0

def _feature_error(pred, actual):
    resid = [p - a for p, a in zip(pred, actual)]
    rms = _rms(resid)
    max_abs = max(abs(r) for r in resid)
    return _RMS_WEIGHT * saturate(rms, _SAT_FEATURE) + _MAX_WEIGHT * saturate(max_abs, _SAT_FEATURE)
```

- `resid` = per-feature prediction errors (predicted minus actual).
- `_rms` = **Root Mean Square** = √(average of squared errors) — the typical
  error size across all features.
- `max_abs` = the single **worst** feature's error.
- We blend them (40% average, 60% worst). Why favour the worst? Because an attack
  often spikes *one* feature enormously (e.g. `failed_connections`) while the rest
  stay normal — a pure average would dilute that one spike among seven calm
  numbers and hide it. Weighting the max keeps the signal sharp. This exact
  reasoning is why detection works well.

### Node-state error

```python
def _node_state_error(pred, actual):
    return saturate(euclidean(pred, actual), _SAT_NODE) if pred else 0.0
```

- The overall straight-line (**Euclidean**) distance between predicted and actual
  feature vectors, saturated to [0, 1). A second, holistic view of "how far off."

### Structural error (Jaccard) and edge-state error

```python
def _structural_error(predicted, actual):
    pe, ae = predicted.edge_set(), actual.edge_set()
    union = pe | ae
    return 1.0 - (len(pe & ae) / len(union)) if union else 0.0
```

- **Jaccard distance** on the edge sets (from concepts): `1 − (shared ÷ total)`.
  `pe & ae` is edges in *both* (intersection); `pe | ae` is edges in *either*
  (union). If the predicted and actual connections match perfectly → 0; totally
  different → 1. Detects unexpected *structural* changes (new connections
  appearing).

```python
def _edge_state_error(predicted, actual):
    pw = {(e.src,e.dst): e.weight for e in predicted.edges}
    aw = {(e.src,e.dst): e.weight for e in actual.edges}
    keys = set(pw) | set(aw)
    max_w = max([abs(v) for v in list(pw.values())+list(aw.values())] + [1.0])
    total = sum(abs(pw.get(k,0.0) - aw.get(k,0.0)) / max_w for k in keys)
    return min(1.0, total / len(keys)) if keys else 0.0
```

- Compares edge **weights** (traffic volumes) between prediction and reality,
  normalised by the biggest weight so it stays in [0, 1]. Catches "same
  connections, but wildly different traffic."

### Putting it together: `compute_deviation`

```python
def compute_deviation(predicted, actual, previous=None, weights=None,
                      anomaly_threshold=0.55, deviating_threshold=0.35):
    w = weights or DEFAULT_WEIGHTS
    structural_error = _structural_error(predicted, actual)
    edge_state_error = _edge_state_error(predicted, actual)

    per_node = {}
    scored_keys = set(actual.nodes) & set(predicted.nodes)
    for key in scored_keys:
        p_feat = predicted.nodes[key].features
        a_feat = actual.nodes[key].features
        feature_err    = _feature_error(p_feat, a_feat)
        node_state_err = _node_state_error(p_feat, a_feat)

        temporal_err = 0.0
        if previous is not None and key in previous.nodes:
            prev = previous.nodes[key].features
            actual_delta = euclidean(a_feat, prev)   # how much it REALLY moved
            pred_delta   = euclidean(p_feat, prev)    # how much we PREDICTED it moves
            temporal_err = saturate(abs(actual_delta - pred_delta), _SAT_TEMPORAL)

        score = (w["feature"]*feature_err + w["node_state"]*node_state_err
                 + w["structural"]*structural_error + w["edge_state"]*edge_state_error
                 + w["temporal"]*temporal_err)
        score = max(0.0, min(1.0, score))
        status = ("anomalous" if score >= anomaly_threshold
                  else "deviating" if score >= deviating_threshold
                  else "normal")
        per_node[key] = NodeDeviation(key, score, feature_err, node_state_err,
                                      structural_error, edge_state_error,
                                      temporal_err, status)

    graph_score = (sum(d.deviation_score for d in per_node.values())/len(per_node)
                   if per_node else 0.0)
    return DeviationResult(actual.index, graph_score, structural_error,
                           edge_state_error, per_node)
```

Walkthrough:
- We only score nodes present in **both** the prediction and reality
  (`set(actual.nodes) & set(predicted.nodes)`).
- For each node we compute the three per-node errors. **temporal error** is
  subtle: it compares *how much the node actually changed* since the previous
  window vs *how much we predicted it would change*. If reality lurched but we
  predicted calm (or vice versa), temporal error rises. It needs the `previous`
  graph, so it's skipped for the very first window.
- The five components are combined with the weights into `score`, clamped to
  [0, 1], then turned into a **status** by two thresholds: ≥ anomaly → "anomalous",
  ≥ deviating → "deviating", else "normal". (The pipeline passes the calibrated
  thresholds 0.45 / 0.25; the defaults here are just fallbacks.)
- `graph_score` is the average node score — a single "how weird is this whole
  window" number.
- `NodeDeviation` and `DeviationResult` are simple `@dataclass` records holding
  these numbers; `DeviationResult` also offers `anomalous_keys()` /
  `deviating_keys()` helpers.

That's the entire anomaly brain: forecast error, decomposed, weighted,
thresholded.

---

## E.2 `forecast/engine.py` — driving the model over time

```python
class ForecastEngine:
    def __init__(self, model, weights=None,
                 anomaly_threshold=0.55, deviating_threshold=0.35):
        self.model = model
        self.weights = weights
        self.anomaly_threshold = anomaly_threshold
        self.deviating_threshold = deviating_threshold
```

- A thin wrapper bundling a fitted `model` with the scoring settings. It offers
  the operations the rest of the system needs.

```python
    def forecast(self, history, k, dropout=0.0, rng=None):
        if k < 1:
            raise ValueError("Forecast horizon k must be >= 1")
        return self.model.predict_sequence(history, k, dropout=dropout, rng=rng)
```

- `forecast` — produce the K-step look-ahead for the dashboard's Forecast view.
  It just delegates to the model's `predict_sequence` (the autoregressive rollout
  from Part D).

```python
    def score_window(self, graphs, t):
        if t < 1 or t >= len(graphs):
            raise IndexError("score_window requires 1 <= t < len(graphs)")
        predicted = self.model.predict_next(graphs[:t])       # predict window t
        return compute_deviation(predicted, actual=graphs[t], previous=graphs[t-1],
                                 weights=self.weights,
                                 anomaly_threshold=self.anomaly_threshold,
                                 deviating_threshold=self.deviating_threshold)
```

- `score_window(graphs, t)` — the causal heart. To score window `t`, it predicts
  it using **only** the earlier windows `graphs[:t]` (the `:t` slice excludes `t`),
  then compares against the real `graphs[t]`. Because prediction uses only the
  past, there is **no future leakage** — this is honest, streaming-style scoring.

```python
    def rolling_deviation(self, graphs):
        return [self.score_window(graphs, t) for t in range(1, len(graphs))]
```

- Run `score_window` across the whole sequence (from window 1 onward; window 0
  has no past to predict from). Returns one `DeviationResult` per window — the
  full timeline of "how weird was each moment."

```python
    @staticmethod
    def apply_statuses(graph, result):
        for key, node in graph.nodes.items():
            dev = result.per_node.get(key)
            node.status = dev.status if dev else "normal"
```

- Writes each node's computed status ("anomalous" etc.) back onto the graph, so
  the dashboard can colour nodes and the propagation engine can see which nodes
  are "infected."

---

## Recap

Step 4 done. `compute_deviation` turns "predicted graph vs real graph" into an
interpretable 0–1 score per node (feature + node + structural + edge + temporal
errors, blended and saturated), then labels each node normal/deviating/anomalous.
`ForecastEngine` runs this causally across the whole timeline (no leakage) and
also produces K-step forecasts. This is the detection core the analytics layer
builds on.

Next: [Code F — the analytics engines](06-analytics.md)
