# 4 · Architecture & Data Flow

Now we zoom out and see how all the folders fit together. Keep the "pipe" mental
model from Part 0 handy — this part makes it concrete.

---

## 4.1 The folder map (what lives where)

```
SIH-Rishi-/
├── sentinelx/                ← the Python package (all the brains)
│   ├── config.py             ← settings + our tiny YAML reader
│   ├── seeding.py            ← reproducible randomness
│   ├── linalg.py             ← hand-written matrix maths
│   ├── data/                 ← STEP 1: get + clean + featurise traffic
│   │   ├── schema.py         ← the FlowRecord (one network flow)
│   │   ├── synthetic.py      ← fake-but-realistic traffic generator
│   │   ├── preprocess.py     ← CIC-IDS2018 CSV loader + Normalizer (z-score)
│   │   ├── features.py       ← windowing + per-node feature vectors
│   │   └── splits.py         ← leakage-safe train/test split
│   ├── graph/                ← STEP 2: build the dynamic graph
│   │   ├── types.py          ← NodeState, EdgeState, GraphState
│   │   └── builder.py        ← turn a window of flows into a GraphState
│   ├── models/               ← STEP 3: the world model (learn "normal")
│   │   ├── base.py           ← the WorldModel interface + dropout helper
│   │   ├── statistical.py    ← Persistence + EWMA baselines
│   │   ├── linear.py         ← ridge-regression transition model
│   │   └── registry.py       ← build a model by name
│   ├── forecast/             ← STEP 4: predict + score deviation
│   │   ├── deviation.py      ← the behavioural deviation score
│   │   └── engine.py         ← K-step forecast + rolling scoring
│   ├── analytics/            ← STEP 5: the "four questions" engines
│   │   ├── propagation.py    ← infection spread + Rₑ
│   │   ├── uncertainty.py    ← MC-Dropout confidence
│   │   ├── novelty.py        ← OOD / KNOWN→UNKNOWN
│   │   ├── stability.py      ← perturbation robustness
│   │   ├── counterfactual.py ← what-if interventions
│   │   ├── mitre.py          ← ATT&CK stage mapping
│   │   └── explain.py        ← per-feature contributions
│   ├── persistence/          ← STEP 6: store everything
│   │   ├── schema.sql        ← the database tables
│   │   ├── db.py             ← SQLite connection (with safe fallback)
│   │   └── repository.py     ← read/write helpers
│   ├── narrative/            ← STEP 7: plain-English incident log
│   │   └── chronicle.py      ← templated sentences (CyberChronicle)
│   ├── pipeline.py           ← THE ORCHESTRATOR: runs steps 1→7 in order
│   ├── api/server.py         ← STEP 8: HTTP API + serves the dashboard
│   └── cli.py                ← command-line entry point (run / serve / models)
├── frontend/                 ← the dashboard (HTML + CSS + JS + D3)
├── tests/                    ← the safety net (pytest)
├── configs/                  ← YAML experiment settings
└── docs/                     ← this book
```

Notice the folders mirror the pipeline stages 1→8. That's not an accident — the
architecture *is* the pipeline.

---

## 4.2 The data flow (end to end)

Here is exactly what happens on one run, following a drop of data through the
pipe:

```
                 configs/*.yaml  +  seed
                          │
                          ▼
          ┌───────────────────────────────┐
          │  data/                        │   1. get raw flows (synthetic or CSV)
          │  synthetic / CIC loader       │   2. clean them (drop junk)
          │  features (windows + vectors) │   3. cut into time windows,
          └───────────────┬───────────────┘      compute node feature vectors
                          ▼
          ┌───────────────────────────────┐
          │  graph/builder                │   4. each window → a GraphState
          └───────────────┬───────────────┘      (nodes, edges, features)
                          ▼
          ┌───────────────────────────────┐
          │  data/splits  (temporal)      │   5. split: early windows = train,
          └───────────────┬───────────────┘      later windows = test (no leak)
                          ▼
          ┌───────────────────────────────┐
          │  Normalizer.fit(TRAIN only)   │   6. z-score using TRAIN stats,
          │  then transform ALL graphs    │      apply to every graph
          └───────────────┬───────────────┘
                          ▼
          ┌───────────────────────────────┐
          │  models/  model.fit(TRAIN)    │   7. learn "normal" dynamics
          │  novelty scorer.fit(TRAIN)    │
          └───────────────┬───────────────┘
                          ▼
          ┌───────────────────────────────┐
          │  forecast/engine              │   8. for each window: predict it
          │  rolling_deviation            │      from the past, score the gap →
          └───────────────┬───────────────┘      mark nodes normal/deviating/anomalous
                          ▼
          ┌───────────────────────────────┐
          │  analytics/                   │   9. propagation, uncertainty,
          │  (the four-questions engines) │      novelty, stability, counterfactual
          └───────────────┬───────────────┘
                          ▼
          ┌───────────────────────────────┐
          │  narrative + persistence      │  10. write incidents in English;
          │  (CyberChronicle → SQLite)    │      save everything to the DB
          └───────────────┬───────────────┘
                          ▼
          ┌───────────────────────────────┐
          │  api/server  →  frontend      │  11. serve results as JSON +
          │  (REST + D3 dashboard)        │      draw them in the browser
          └───────────────────────────────┘
```

`pipeline.py` is the conductor that calls steps 1–10 and hands the result to the
API. When you run `python -m sentinelx.cli run`, steps 1–10 happen and print a
summary. When you run `... serve`, step 11 additionally starts the web server.

---

## 4.3 Two objects you'll meet everywhere

Almost all the code passes around two things — learn them now:

### `GraphState`
One snapshot of the network for one time window. It holds:
- `nodes`: a dictionary of `node_key → NodeState` (each node's features + status),
- `edges`: a list of `EdgeState` (who talked to whom, with traffic features),
- the feature names, the window index, and the start/end times.

Think of it as "the network, frozen, for this minute."

### `SentinelService`
The result of a full pipeline run, kept in memory. It holds the trained model,
all the `GraphState`s, the deviation results, and the database handle. The API
asks *it* questions like `network_state(window=39)` or
`forecast(window=35, k=3)`. It's the bridge between "the pipeline computed
things" and "the web serves things."

---

## 4.4 The request lifecycle (when the dashboard loads)

When you open the dashboard and click the **Network** tab:

```
browser  ──GET /network/state?window=39──▶  http.server (api/server.py)
                                              │  routes the URL to a function
                                              ▼
                                   service.network_state(39)
                                              │  reads the GraphState + deviations
                                              ▼
                                   builds a JSON dict
browser  ◀────────  JSON  ────────────────────┘
   │
   ▼  app.js draws nodes/edges with D3, colours them by status
 you see the live graph
```

Every screen works this way: the JavaScript asks a URL, the server answers with
JSON, D3 draws it. Simple and uniform.

---

## 4.5 Why in-memory model + SQLite storage (both)?

- The **model and graphs stay in memory** so the API can answer "what-if"
  questions *live* (e.g. run a fresh counterfactual on demand).
- The **database (SQLite)** stores the computed results (snapshots, anomalies,
  incidents) so they can be queried later and survive as a record. It also
  matches the professional pattern where API and storage don't share in-memory
  state.

---

You now have the whole map in your head. From here on, we open each file and read
it, in pipeline order. Start with the foundation.

Next: [Part 5 — Project setup](05-project-setup.md) → then the
[code walkthroughs](code/01-foundation.md).
