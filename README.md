# Sentinel-X

### Predictive Network Behaviour & Threat Forecasting Platform

Most intrusion detection tells you an attack *already happened*. Sentinel-X models
the network as a **living, evolving graph**, learns how it normally behaves over
time, and **forecasts where it is heading** — then quantifies how much to trust
that forecast.

For any moment it answers four questions at once:

1. What will the network look like in the next few time steps?
2. How confident is that forecast?
3. Has this kind of behaviour ever been seen before?
4. If we intervened right now (isolated a host, blocked a port), would the
   predicted future improve?

> **Guiding principle: honest before impressive.** Forecast horizons are extended
> only when validated, propagation metrics are kept only where they measurably
> help, and no generated text is treated as a ground-truth security judgment.

---

## 📚 Learn the whole thing from zero

New to graphs, machine learning, or web development? Start with the
**[complete learning documentation in `docs/`](docs/README.md)** — a
teach-from-scratch handbook that explains every concept, every technology, and
every file line by line, so a beginner can read it once and rebuild the project
themselves. Deployment steps live in **[DEPLOY.md](DEPLOY.md)**.

---

## ⚠️ Read this first: what runs now vs. the framework swap-in

This repository was built and verified inside an **offline sandbox** with no
package-manager access. PyTorch, PyTorch Geometric, FastAPI, pandas, NetworkX,
React/D3/Vite — none of them could be installed or executed here, so claiming a
"bug-free" build on that stack would be unverifiable.

Instead, the entire Sentinel-X architecture (all 14 conceptual layers) is
implemented as a **complete, runnable, fully-tested reference using the Python
standard library only** (plus `sqlite3`, and D3 loaded from a CDN in *your*
browser). Every layer is exercised by a passing `pytest` suite — including the
critical data-leakage tests.

The heavyweight research/production stack is the documented **swap-in**: it slots
in behind the same interfaces without a redesign.

| Layer | Runs now (this repo, zero deps) | Framework swap-in (`requirements-full.txt`) |
|---|---|---|
| Numerics | pure-python `linalg` (`ridge_fit`, `solve`) | NumPy / SciPy |
| World model | statistical + ridge linear transition | PyTorch + PyTorch-Geometric (GraphSAGE+GRU → temporal GNN) |
| Graphs | `sentinelx.graph` dataclasses | NetworkX + PyG |
| Data | stdlib `csv` + synthetic generator | pandas + Scapy (PCAP) |
| Config/tracking | zero-dep YAML subset parser | ruamel.yaml + MLflow |
| API | stdlib `http.server` | FastAPI + uvicorn |
| Dashboard | static HTML + vanilla JS + D3 | React + TypeScript + Vite + Tailwind |
| Persistence | SQLite | Postgres |

The interfaces (`WorldModel`, `ForecastEngine`, the REST contract, the DB schema)
are identical, so swapping a layer is a drop-in, not a rewrite.

---

## Quick start

No installation required — it's the standard library.

```bash
# 1. Run the full pipeline and print a JSON summary (synthetic data)
python -m sentinelx.cli run

# 2. Run a specific config / model / seed
python -m sentinelx.cli run --config configs/default.yaml --model ewma --seed 7

# 3. Launch the dashboard + API, then open the printed URL
python -m sentinelx.cli serve
#   -> Sentinel-X dashboard + API listening on http://127.0.0.1:8787

# 4. Run the test suite (requires pytest)
python -m pytest
```

Example summary from `run` (default synthetic lateral-movement scenario):

```json
{
  "num_windows": 40, "train_windows": 28, "test_windows": 12,
  "model_type": "linear_transition",
  "detection": { "true_positives": 21, "false_positives": 0,
                 "false_negatives": 0, "precision": 1.0, "recall": 1.0 },
  "total_incidents": 13
}
```

---

## How it works

```
config + seed
  → flows (synthetic generator, or a CIC-IDS2018 CSV)
  → clean → per-window dynamic graphs  G_t = (V_t, E_t, X_t)
  → leakage-safe temporal split (train = past, test = strictly-later future)
  → fit z-score normaliser + world model + novelty scorer  (TRAIN ONLY)
  → rolling one-step forecast + behavioural deviation score  D_t = d(G_t, Ĝ_t)
  → propagation · uncertainty · novelty · stability · counterfactuals
  → SQLite persistence + templated CyberChronicle incidents
  → HTTP API + dashboard
```

### The layers

- **Dynamic graph** — each time window is a graph; node identity is a stable
  hashed key so nodes align across windows even as the vertex set changes.
- **World model** — learns `p(G_{t+1} | G_{≤t})`. Progressive complexity:
  persistence → EWMA → ridge linear transition (→ GraphSAGE+GRU in the full
  stack). All behind one `WorldModel` interface.
- **Behavioural deviation** — decomposes forecast error into feature, node-state,
  structural, edge-state and temporal components, bounded to `[0, 1]` and
  thresholded into `normal / deviating / anomalous`.
- **Propagation (cyber-epidemiology)** — velocity, intensity and an effective
  reproduction number Rₑ as an infection spreads host→host.
- **Uncertainty** — MC-Dropout: N stochastic passes → mean/σ → LOW/MEDIUM/HIGH,
  growing with the forecast horizon.
- **Novelty / OOD** — embedding distance + prediction error + uncertainty →
  `KNOWN … UNKNOWN`.
- **Stability** — perturb the evidence, re-forecast, measure the swing → STABLE /
  UNSTABLE.
- **Counterfactuals** — `ISOLATE_NODE`, `BLOCK_EDGE`, `BLOCK_PORT`,
  `DISABLE_COMMUNICATION`, `RATE_LIMIT` over the world model: `ΔRisk = before − after`.
- **MITRE ATT&CK** + **explainability** — post-hoc interpretability only, with the
  caveat that attributions are indicative, not proof.
- **CyberChronicle** — deterministic templated incident narration.

### Two design decisions worth calling out

- **Z-score, not min-max, normalisation.** Min-max scaling fit on benign data
  *clamps* an attack's out-of-range spikes to 1.0 — the same value as the benign
  maximum — erasing the signal. Z-scoring *amplifies* out-of-distribution values,
  so forecast error explodes on anomalies. (Covered by a leakage test.)
- **Grid-aligned time windows.** Windows anchor to a `window_seconds` grid so a
  data source emitting on a `w · window_seconds` schedule maps 1:1 onto window
  indices — no silent off-by-one between the generator and the graph builder.

---

## API

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Liveness |
| `/summary` | GET | Run metadata + detection metrics |
| `/network/state?window=<i>` | GET | Graph snapshot + per-node deviations |
| `/forecast?window=<i>&k=<n>` | GET | K-step forecast with per-step uncertainty |
| `/uncertainty?window=<i>` | GET | MC-Dropout mean/σ + label |
| `/propagation` | GET | Propagation events + Rₑ |
| `/counterfactual` | POST | Simulate an intervention (`ΔRisk`) |
| `/ingest` | POST | Re-run the pipeline with new parameters |
| `/incident` | GET | CyberChronicle narrative log |

---

## Dashboard

A dark "command-center" single-page dashboard (`frontend/`) served by the API:
**Network** (force-directed live graph), **Forecast** (horizon scrubber),
**Uncertainty** (confidence band), **Propagation** (infection path + metrics),
**Counterfactual** (before/after risk table), and **CyberChronicle** (incident
feed). D3 is loaded from a CDN; everything else is dependency-free.

---

## Project layout

```
sentinelx/
  config.py          reproducible config + zero-dep YAML subset parser
  seeding.py         deterministic RNG streams
  linalg.py          pure-python matrix ops + ridge regression
  data/              synthetic gen, CIC-IDS2018 loader, features, temporal splits
  graph/             GraphState + dynamic graph builder
  models/            WorldModel interface + statistical/linear models + registry
  forecast/          K-step engine + behavioural deviation scoring
  analytics/         propagation, uncertainty, novelty, stability, counterfactual,
                     mitre, explain
  persistence/       SQLite schema + repository
  narrative/         templated CyberChronicle
  pipeline.py        end-to-end orchestration + SentinelService
  api/               stdlib HTTP API + static dashboard host
  cli.py             `sentinelx run | serve | models`
configs/             reproducible YAML experiment configs
frontend/            static D3 dashboard
tests/               pytest suite incl. data-leakage tests
requirements-full.txt  the framework swap-in (torch, PyG, FastAPI, ...)
```

## Reproducibility & anti-leakage

- Every run is defined by `config + seed`; the config snapshot is stored in SQLite.
- Splits are **strictly chronological**; `assert_no_leakage` rejects any overlap,
  time-travel, or unordered split.
- The feature scaler is fit on **training windows only** and cannot `transform`
  before `fit`. Both are enforced by tests in `tests/test_leakage.py`.

## Testing

```bash
python -m pytest          # 73 tests, ~3s, standard library + pytest only
```

## License

MIT.
