# 7 · Glossary

Every term used in this book, in one place. Skim it, or use it as a lookup.

### Core idea
- **Graph** — a set of things (**nodes**) connected by links (**edges**). Here:
  computers connected by network connections.
- **Node / vertex** — one thing in a graph (a host or server).
- **Edge** — a connection between two nodes. **Directed** = it has an arrow
  (who initiated).
- **Adjacency** — the list of a node's direct neighbours.
- **Dynamic graph** — a graph captured as a sequence of snapshots over time.
- **Time window** — a fixed slice of time (e.g. 60s); one snapshot per window.
- **GraphState (Gₜ)** — one window's snapshot: nodes + edges + features.

### Data & features
- **Flow** — one bidirectional conversation between two machines.
- **FlowRecord** — our canonical representation of a flow.
- **Feature** — one measurable number describing behaviour (e.g. unique
  destinations).
- **Feature vector** — the fixed-order list of a node's features.
- **Normalisation** — putting features on a comparable scale.
- **Min–max scaling** — squeeze values to [0, 1]; *clamps* outliers (bad for us).
- **Z-score / standardisation** — measure in standard deviations from the mean;
  *amplifies* outliers (what we use).
- **Mean** — average. **Standard deviation (std)** — typical spread around the
  mean.
- **Data leakage** — future/test info secretly influencing training (cheating).
- **Temporal split** — train on the past, test on the strictly-later future.

### Modelling
- **Model** — a function with tunable knobs that learns a pattern.
- **Training / fitting** — tuning the knobs to match examples.
- **Prediction / inference** — using the tuned model on new input.
- **World model** — a model that predicts the next network state from the past.
- **Baseline** — a deliberately simple model the real one must beat.
- **Persistence model** — predicts "next = current".
- **EWMA** — Exponentially Weighted Moving Average; a smoothed recent average.
- **Weight / bias** — a model's tunable multiplier / constant offset.
- **Linear regression** — fit a straight-line relationship.
- **Ridge regression** — linear regression with a penalty (λ) that keeps weights
  small; more stable.
- **Regularisation** — nudging a model toward simpler solutions.
- **Identity matrix (I)** — the matrix version of the number 1.
- **Transpose (ᵀ)** — flip a matrix's rows and columns.
- **Dot product** — multiply matching elements of two vectors and sum.
- **Matrix multiplication** — many dot products; combines linear maps.
- **Normal equation** — the closed-form solution for linear/ridge regression.
- **Gauss–Jordan elimination** — the systematic method to solve `A·X = B`.

### Forecasting & detection
- **Forecasting** — predicting future values from past ones.
- **K-step forecast** — predicting K windows ahead.
- **Autoregressive rollout** — feed each prediction back to predict the next step.
- **Behavioural deviation (Dₜ)** — the gap between predicted and actual graph;
  the anomaly score.
- **MSE / RMS** — Mean Squared Error / its square root; "how wrong" measures.
- **Euclidean distance** — straight-line distance between two vectors.
- **Jaccard distance** — 1 − (shared ÷ total); compares two sets (here, edge
  sets).
- **Saturating function** — `x/(x+c)`; squashes any value into [0, 1).
- **Threshold** — a cutoff turning a score into a label (normal/deviating/
  anomalous).

### The four questions
- **Uncertainty** — how much to trust a forecast.
- **Dropout** — randomly zeroing inputs; run many times to gauge spread.
- **MC-Dropout (Monte-Carlo)** — estimating uncertainty by many random dropout
  passes.
- **Propagation** — an anomaly spreading along edges, like an infection.
- **Propagation velocity / intensity** — new infections per second / their
  strength.
- **Effective reproduction number (Rₑ)** — new infections per current infection
  (the COVID "R number"). > 1 = growing.
- **Novelty / Out-of-Distribution (OOD)** — behaviour unlike anything seen in
  training.
- **Embedding** — a fixed-length numeric fingerprint of a graph.
- **Stability** — how much a forecast changes under a tiny input **perturbation**
  (nudge).
- **Counterfactual** — a "what if I did X?" simulation.
- **Intervention** — the simulated action (ISOLATE_NODE, BLOCK_PORT, ...).
- **Reachability** — which nodes can be reached by following edges.
- **Risk** — how exposed the servers/estate are to compromised nodes.
- **ΔRisk** — risk before minus risk after an intervention.

### Interpretation
- **MITRE ATT&CK** — an industry catalogue of attacker behaviours by stage.
- **Lateral movement** — an attacker spreading sideways through a network.
- **Exfiltration** — stealing data out of the network.
- **Explainability** — attributing a decision to specific inputs/features.

### Software & tooling
- **Standard library (stdlib)** — modules that ship inside Python; no install.
- **Package / module** — an importable folder / file of Python code.
- **Dataclass** — a class that auto-generates its boilerplate.
- **Type hint** — an annotation like `list[int]`; documentation for humans/tools.
- **Interface / abstract base class (ABC)** — a contract subclasses must fulfil.
- **Polymorphism** — swapping one implementation for another behind a shared
  interface.
- **Decorator** — a `@label` that modifies the function/class below it.
- **Recursion** — a function calling itself (used for nested merges).
- **List/dict/set comprehension** — concise `[expr for x in ...]` builders.
- **f-string** — `f"{value}"` string formatting.
- **YAML** — a human-friendly settings format (indentation + `key: value`).
- **JSON** — a text format for structured data; the API's language.
- **SQL / SQLite** — a query language / a file-based database in stdlib.
- **Primary key / foreign key** — a row's unique id / a link to another table.
- **Parameterised query** — passing values separately from SQL (blocks
  **SQL injection**).
- **HTTP / GET / POST** — the web protocol / "fetch data" / "send data".
- **Status code** — 200 OK, 400 bad request, 404 not found, 500 server error.
- **Route / router** — mapping a URL path to a handler function.
- **CORS** — headers letting a browser call an API across origins.
- **Path traversal** — an attack using `..` to escape a folder (we block it).
- **CDN** — a public host for popular libraries (we load D3 from one).
- **HTML / CSS / JavaScript** — page structure / style / behaviour.
- **D3.js** — a JS library for data-driven graphics (our force graph, charts).
- **Force simulation** — a physics layout for graphs (springs + repulsion).
- **fetch / async / await** — browser HTTP calls and how JS waits for them.

### Practice
- **Seed** — the starting number that makes "random" reproducible.
- **Pseudo-random** — deterministic numbers that look random.
- **Reproducibility** — anyone can re-run and get identical results.
- **Ablation** — removing/simplifying a part to measure its contribution.
- **Fixture** — reusable setup for tests (pytest).
- **Precision / Recall** — of what we flagged, how many were real / of the real
  ones, how many we caught.
- **True/False Positive/Negative (TP/FP/FN/TN)** — the four outcomes of a
  detector.

Back to the [index](README.md).
