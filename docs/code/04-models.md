# Code D · The World Models (`models/`)

Pipeline **step 3**: learn what "normal" looks like, so we can predict the next
window. Four files: `base.py` (the shared interface), `statistical.py` (two dumb
baselines), `linear.py` (the real learner), `registry.py` (build by name).

Recall the definition: a **world model** learns `p(G_{t+1} | G_{≤t})` — "given
the history, what's the next graph?" In this project a "prediction" is a
`GraphState` whose node features are the forecast, and whose edges are carried
over from the last real window (a fair, shared structural baseline).

---

## D.1 `models/base.py` — the interface everyone obeys

### An interface (abstract base class)

```python
from abc import ABC, abstractmethod

class WorldModel(ABC):
    name = "base"

    @abstractmethod
    def fit(self, train_graphs): ...

    @abstractmethod
    def predict_next(self, history, dropout=0.0, rng=None): ...
```

- `ABC` = **Abstract Base Class**. It defines a **contract**: any real model must
  provide `fit` and `predict_next`. `@abstractmethod` means "subclasses *must*
  implement this; you cannot create a bare `WorldModel`."
- Why do this? So the *rest* of the code (forecast engine, uncertainty,
  counterfactual) talks only to `WorldModel` and never cares whether it's the
  persistence baseline or a future GNN. This is **polymorphism** — swap the model,
  everything else is unchanged. It's the single most important design idea for
  making the project extensible.
- `fit(train_graphs)` — learn from the training sequence.
- `predict_next(history, dropout, rng)` — forecast the next graph. The `dropout`
  and `rng` arguments enable the MC-Dropout uncertainty trick (below).

### The dropout helper

```python
def apply_dropout(vec, dropout, rng):
    if dropout <= 0.0 or rng is None:
        return list(vec)
    keep = 1.0 - dropout
    scale = 1.0 / keep if keep > 0 else 0.0
    return [(0.0 if rng.random() < dropout else v * scale) for v in vec]
```

- **Dropout** randomly zeroes each input feature with probability `dropout`.
- **Inverted dropout:** the survivors are scaled up by `1/keep` so the *average*
  magnitude stays the same. (E.g. drop 20% → keep 80% → multiply survivors by
  1.25.) This is the standard trick so dropout doesn't shrink the signal.
- If `dropout` is 0, it just returns a copy (no-op). Running `predict_next` many
  times with dropout on gives slightly different answers each time — that spread
  *is* the uncertainty estimate. Cleverly, this works for *any* model, even the
  dumb baselines.

### Shared helpers on the base class

```python
    def _skeleton_from_last(self, last):
        pred = last.clone()
        pred.index = last.index + 1
        pred.window_start = last.window_end
        pred.window_end = last.window_end + (last.window_end - last.window_start)
        return pred
```

- Builds the *shell* of the prediction by cloning the last real graph (keeping
  its structure/edges) and advancing the time window by one. Subclasses then
  overwrite the node **features** with their forecast. Sharing this means all
  models make the *same* structural assumption, so their *feature* forecasts are
  compared fairly.

```python
    def predict_sequence(self, history, k, dropout=0.0, rng=None):
        rolling = list(history)
        out = []
        for _ in range(k):
            nxt = self.predict_next(rolling, dropout=dropout, rng=rng)
            out.append(nxt)
            rolling = list(rolling) + [nxt]
        return out
```

- **K-step autoregressive rollout** (from concepts): predict one step, append the
  prediction to the history, predict again, `k` times. Each model gets this for
  free from the base class — it only has to implement one-step `predict_next`.

```python
    @staticmethod
    def _set_features(pred, key, features):
        node = pred.nodes.get(key)
        if node is None:
            pred.nodes[key] = NodeState(key=key, label=key, features=features)
        else:
            node.features = features
```

- A small helper to write a forecast vector onto a node in the prediction graph.
  `@staticmethod` means it doesn't need `self` — it's just a utility grouped with
  the class.

---

## D.2 `models/statistical.py` — the honest baselines

### Persistence: "tomorrow = today"

```python
class PersistenceModel(WorldModel):
    name = "baseline_statistical"

    def fit(self, train_graphs):
        return self                      # nothing to learn!

    def predict_next(self, history, dropout=0.0, rng=None):
        last = history[-1]
        pred = self._skeleton_from_last(last)
        for key, node in last.nodes.items():
            self._set_features(pred, key, apply_dropout(node.features, dropout, rng))
        return pred
```

- The simplest possible "model": predict that the next window looks **exactly
  like the last one**. `fit` does nothing (`return self` just hands the object
  back so you can write `model.fit(...).predict(...)`).
- Why keep something this dumb? It's the **bar to beat**. If a fancy model can't
  outperform "assume no change," the fancy model is useless. Baselines keep us
  honest.

### EWMA: a smoothed average

```python
class EWMAModel(WorldModel):
    name = "ewma"
    def __init__(self, alpha=0.4):
        if not (0.0 < alpha <= 1.0):
            raise ValueError("EWMA alpha must be in (0, 1]")
        self.alpha = alpha

    def predict_next(self, history, dropout=0.0, rng=None):
        last = history[-1]
        pred = self._skeleton_from_last(last)
        for key in last.nodes:
            ewma = None
            for g in history:               # walk oldest → newest
                node = g.nodes.get(key)
                if node is None: continue
                if ewma is None:
                    ewma = list(node.features)
                else:
                    ewma = [self.alpha*f + (1-self.alpha)*e
                            for f, e in zip(node.features, ewma)]
            self._set_features(pred, key, apply_dropout(ewma or [0.0]*dim, dropout, rng))
        return pred
```

- **EWMA** = Exponentially Weighted Moving Average. It predicts a *smoothed*
  version of the node's recent history, weighting recent windows more.
- The update `ewma = alpha·new + (1-alpha)·old` is the whole formula. With
  `alpha = 0.4`, each new window contributes 40% and the accumulated past 60%.
  Higher alpha = more reactive; lower = smoother.
- The `__init__` guard rejects an invalid `alpha` immediately — fail fast with a
  clear message.

---

## D.3 `models/linear.py` — the real learner (ridge regression)

This is our main world model. It actually *learns* how features evolve from one
window to the next, using the `ridge_fit` you read in Part A.

```python
class LinearTransitionModel(WorldModel):
    name = "linear_transition"
    def __init__(self, ridge_lambda=0.05):
        self.ridge_lambda = ridge_lambda
        self.W = []            # the learned weight matrix
        self._wt = []          # its transpose (cached for fast prediction)
        self.in_dim = 0
        self.out_dim = 0
        self._fitted = False
```

- `ridge_lambda` is the regularisation strength λ from concepts (keeps weights
  modest). `W` will hold the learned mapping "features now → features next."

### Training

```python
    def fit(self, train_graphs):
        xs, ys = [], []
        for g_t, g_next in zip(train_graphs, list(train_graphs)[1:]):
            for key, node in g_t.nodes.items():
                nxt = g_next.nodes.get(key)
                if nxt is None: continue
                xs.append(list(node.features) + [1.0])   # + bias term
                ys.append(list(nxt.features))
        if len(xs) < 2:
            self._fitted = False
            return self
        self.in_dim = len(xs[0]); self.out_dim = len(ys[0])
        self.W = ridge_fit(xs, ys, self.ridge_lambda)
        self._wt = transpose(self.W)
        self._fitted = True
        return self
```

Line by line — this is the heart of "learning":
- `zip(train_graphs, train_graphs[1:])` pairs each window with the **next** one:
  `(G₁,G₂), (G₂,G₃), ...`. That's how we build "input → next-output" examples.
- For every node present in *both* consecutive windows, we make a training pair:
  input = its features now (`xs`), target = its features next window (`ys`).
- `+ [1.0]` appends a **bias term** (the constant "intercept" from concepts). By
  adding a fixed 1.0 column, the learned weights automatically include an offset.
- Learning across *all nodes* (not one model per node) means the model
  generalises to nodes it never saw individually — essential because the set of
  machines changes every window.
- If there aren't enough pairs, it marks itself unfitted and will fall back to
  persistence (graceful degradation).
- `self.W = ridge_fit(xs, ys, λ)` — the one line that does the actual training,
  using the hand-written maths from Part A. `self._wt = transpose(self.W)` caches
  the transpose so prediction is a fast `matvec`.

### Prediction

```python
    def predict_next(self, history, dropout=0.0, rng=None):
        last = history[-1]
        pred = self._skeleton_from_last(last)
        for key, node in last.nodes.items():
            feats = apply_dropout(node.features, dropout, rng)
            if self._fitted and len(feats) + 1 == self.in_dim:
                x = list(feats) + [1.0]
                y = matvec(self._wt, x)       # x · W  → predicted next features
            else:
                y = list(feats)               # fallback: persistence
            self._set_features(pred, key, y)
        return pred
```

- For each node in the last window: optionally apply dropout, append the bias 1.0,
  then compute `y = x · W` via `matvec`. `y` is the predicted next-window feature
  vector. If the model isn't fitted, it falls back to copying the current features
  (persistence). Robust by design.

> **This is the "GraphSAGE+GRU / temporal-GNN" slot.** The interface is identical;
> a neural version would replace `ridge_fit`/`matvec` with a trained network. We
> chose linear because it trains instantly, is fully understandable, and gives an
> honest baseline the GNN must beat.

---

## D.4 `models/registry.py` — build a model by name

```python
_ALIASES = {"baseline_statistical": "persistence", "statistical": "persistence"}

def available_models():
    return ["persistence", "ewma", "linear_transition"]

def build_model(model_type, model_cfg=None):
    cfg = model_cfg or {}
    key = _ALIASES.get(model_type, model_type)
    if key == "persistence":       return PersistenceModel()
    if key == "ewma":              return EWMAModel(alpha=float(cfg.get("ewma_alpha", 0.4)))
    if key == "linear_transition": return LinearTransitionModel(ridge_lambda=float(cfg.get("ridge_lambda", 0.05)))
    raise ValueError(f"Unknown model_type={model_type!r}. Available: {available_models()} ...")
```

- A **registry** maps a string (from the config, e.g. `"linear_transition"`) to
  an actual model object, passing the right settings. This is how
  `experiment.model_type` in the YAML picks the model without any `if` statements
  scattered across the codebase.
- `_ALIASES` lets a couple of names point to the same model.
- Unknown names raise a helpful error listing the valid options (and noting GNNs
  need the full stack). Fail loudly, never silently.

---

## Recap

Step 3 done. Every model implements the same `WorldModel` interface
(`fit` + `predict_next`), so the rest of the system is model-agnostic. We have two
honest baselines (persistence, EWMA) and one real learner (ridge linear
transition) that trains via the `ridge_fit` maths from Part A. MC-Dropout support
is baked into `predict_next` for free. A registry builds any of them by name from
config.

Next: [Code E — forecasting & the deviation score](05-forecast.md)
