# Code H · Narrative, Pipeline & CLI

Pipeline **step 7** (write the story) and the **orchestrator** that runs steps
1–10 in order, plus the **command-line** entry point.

---

## H.1 `narrative/chronicle.py` — CyberChronicle (plain-English log)

```python
def _clock(window_start):
    total = int(window_start)
    return f"{(total // 3600) % 24:02d}:{(total // 60) % 60:02d}"
```

- Turns a window's start time (in seconds) into a `HH:MM` clock string.
  `//` is integer division, `%` is remainder. `:02d` formats a number with a
  leading zero (so `6` → `06`). Example: 1800 seconds → `00:30`.

```python
def narrate_deviation(window_start, node_key, score, top_features, mitre_stage):
    feats = ", ".join(top_features) if top_features else "multiple signals"
    return (f"{_clock(window_start)} — Behavioral deviation detected on {node_key}. "
            f"Observed network state diverged from the learned baseline "
            f"(deviation {score:.0%}); dominant signals: {feats}. "
            f"Mapped ATT&CK stage: {mitre_stage}.")
```

- Builds one **templated sentence** from structured inputs. `{score:.0%}` formats
  a fraction as a percentage (`0.84` → `84%`). `", ".join(list)` glues a list into
  `"a, b, c"`.
- There are four such templates: `narrate_deviation`, `narrate_propagation`
  (A→B with Rₑ), `narrate_forecast` (trajectory risk), `narrate_intervention`
  (before→after risk).

**Why templates and not an LLM?** This is a deliberate honesty choice from the
guiding principle: the narration is *deterministic* — same inputs always give the
same, auditable sentence. No AI-generated prose is treated as a security
judgement. (An LLM summariser could sit *on top* of this later, but never
*replace* the structured signals.)

---

## H.2 `pipeline.py` — the orchestrator

This file wires all the layers into one reproducible run and exposes a live
service for the API.

### `run_pipeline` — the conductor

```python
def run_pipeline(config_path=None, overrides=None, db_path=None, reset_db=True):
    cfg = load_config(config_path, overrides)
    window_seconds = float(cfg.get("data.window_seconds", 60))

    flows = clean_flows(_load_flows(cfg))                          # step 1
    graphs = build_graph_sequence(flows, int(window_seconds),
                                  float(cfg.get("graph.min_edge_weight", 1.0)))   # step 2
    if len(graphs) < 3:
        raise ValueError("Need at least 3 graph windows to run the pipeline")

    split = temporal_split(len(graphs), float(cfg.get("data.test_fraction", 0.3)))
    assert_no_leakage(split)                                        # step 5 (guard)

    train_rows = []
    for i in split.train_indices:
        train_rows.extend(graphs[i].feature_matrix())
    normalizer = Normalizer(mode="zscore").fit(train_rows)          # step 6: fit on TRAIN only
    for g in graphs:
        for node in g.nodes.values():
            node.features = normalizer.transform_vector(node.features)  # apply to all

    train_graphs = [graphs[i] for i in split.train_indices]
    model = build_model(cfg.get("experiment.model_type", ...), cfg.section("model")).fit(train_graphs)  # step 3
    novelty = NoveltyScorer(**cfg.section("novelty")).fit(train_graphs)

    engine = ForecastEngine(model, weights=dev_cfg.get("weights"),
                            anomaly_threshold=..., deviating_threshold=...)
    devs = engine.rolling_deviation(graphs)                         # step 4
    dev_by_index = {d.graph_index: d for d in devs}
    for d in devs:
        engine.apply_statuses(graphs[d.graph_index], d)             # colour the nodes
```

Read this as the pipeline diagram from Part 4, in code. The **order is the whole
point**, and it encodes the anti-leakage discipline:
1. Load config; get flows; build the graph sequence.
2. Require at least 3 windows (need past → present → future).
3. Split chronologically and **assert no leakage** (from `splits.py`).
4. **Fit the normaliser on training rows ONLY**, then transform *all* graphs with
   those training statistics. This is the exact ordering that prevents the
   "scaler leak." Doing it any other way would be cheating.
5. Fit the model and the novelty scorer on the **training** graphs only.
6. Build the engine with calibrated thresholds; run rolling deviation over the
   whole sequence; write statuses back onto the graphs.

```python
    db = Database(db_path or cfg.get("persistence.db_path", "sentinelx.db"))
    if reset_db: db.reset()
    repo = Repository(db)
    experiment_id = repo.save_experiment(name=..., dataset=..., model_type=...,
                                         config_yaml=cfg.to_yaml(), seed=cfg.seed)
    service = SentinelService(config=cfg, experiment_id=experiment_id, graphs=graphs,
                              model=model, engine=engine, novelty=novelty,
                              normalizer=normalizer, devs=devs, dev_by_index=dev_by_index,
                              window_seconds=window_seconds, summary={}, repo=repo,
                              feature_names=graphs[0].node_feature_names)
    _persist_run(service, split)
    service.summary = _evaluate(service, flows, split)
    return service
```

- Open the database, save the experiment (with the full config for
  reproducibility), then bundle *everything* into a `SentinelService` and persist
  the detailed results. Finally compute the evaluation summary and return the
  service.

### `_persist_run` — save results + write incidents

```python
def _persist_run(service, split):
    ...
    for g in graphs:
        snapshot_ids[g.index] = repo.save_snapshot(experiment_id, g)

    for d in service.devs:
        for nd in d.per_node.values():
            repo.save_anomaly(sid, nd)
        top = max(d.per_node.values(), key=lambda x: x.deviation_score, default=None)
        if top and top.status == "anomalous":
            # explain WHY, name the stage, write a sentence
            tops = top_feature_names(pred.nodes[top.key].features,
                                     actual.nodes[top.key].features, feature_names, 3)
            stage = map_mitre_stage(tops)
            repo.save_incident(sid, "deviation_detected",
                               narrate_deviation(..., top.key, top.deviation_score, tops, stage),
                               mitre_stage=stage, contributing_features=tops)

    events = compute_propagation(graphs, service.dev_by_index, service.window_seconds)
    for e in events:
        repo.save_propagation_event(sid, e)
        repo.save_incident(sid, "propagation_detected", narrate_propagation(..., e))

    for g in graphs:                                # K-step forecasts + confidence
        preds = service.engine.forecast(history, horizon)
        unc  = estimate_uncertainty(model, history, ...)
        nov  = service.novelty.score(g, dev_score, unc.std_dev)
        stab = assess_stability(model, history, ...)
        for step, pg in enumerate(preds, start=1):
            fid = repo.save_forecast(experiment_id, sid, step, pg.to_json())
            if step == 1:
                repo.save_forecast_result(fid, unc, nov, stab)
```

- Saves every snapshot, every node's anomaly scores, and — only for genuinely
  **anomalous** top nodes — writes an incident that combines the three
  interpretation layers: `explain` (which features), `mitre` (which stage),
  `chronicle` (the sentence). Restricting incidents to *anomalous* (not merely
  *deviating*) keeps the log precise and matches the perfect-precision detection.
- Saves propagation events (with narration) and, for each window, the K-step
  forecast plus its uncertainty/novelty/stability. This is the full record the
  dashboard reads.

### `_evaluate` — honest scoring

```python
def _evaluate(service, flows, split):
    truth = _attack_nodes_by_window(flows, service.window_seconds)   # from labels
    tp = fp = fn = 0
    for d in service.devs:
        attack_set = truth.get(d.graph_index, set())
        flagged = set(d.anomalous_keys())
        tp += len(flagged & attack_set)
        fp += len(flagged - attack_set)
        fn += len(attack_set - flagged)
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall    = tp / (tp + fn) if (tp + fn) else 1.0
    return {"detection": {"precision": round(precision, 3), "recall": round(recall, 3), ...}, ...}
```

- Compares the nodes we **flagged anomalous** against the ground-truth attackers
  (from the labels — used *only here*, for scoring, never for training).
- **Precision** = of the ones we flagged, how many were really attacks
  (`tp / (tp+fp)`). **Recall** = of the real attacks, how many we caught
  (`tp / (tp+fn)`). `tp/fp/fn` = true positives / false positives / false
  negatives. On the default synthetic lateral-movement run this comes out
  1.0 / 1.0.
- `_attack_nodes_by_window` uses the **same grid anchoring** as `build_windows`
  (`t_min = (min_ts // ws) * ws`) so ground-truth windows line up exactly with
  graph windows — the off-by-one fix from Part B, applied consistently.

### `SentinelService` — the live query object

```python
@dataclass
class SentinelService:
    config: Config; experiment_id: int; graphs: list[GraphState]
    model: WorldModel; engine: ForecastEngine; novelty: NoveltyScorer
    normalizer: Normalizer; devs: list; dev_by_index: dict
    window_seconds: float; summary: dict; repo: Repository; feature_names: list

    def network_state(self, window=None): ...      # graph JSON + per-node deviations
    def forecast(self, window=None, k=None): ...    # K-step + growing uncertainty
    def uncertainty(self, window=None): ...
    def propagation(self): ...
    def counterfactual(self, action_type, window=None, target_node=None, ...): ...
    def incidents(self): ...
    def _resolve_window(self, window):              # None → last window
        return len(self.graphs)-1 if window is None else window
```

- This object is what the API talks to. It keeps the trained model and graphs in
  memory so it can answer *live* — e.g. `counterfactual(...)` runs a brand-new
  what-if simulation on request, and `forecast(...)` scales uncertainty up with
  `sqrt(step)` so confidence bands widen further out. `_resolve_window(None)`
  defaults to the latest window. Out-of-range windows raise `IndexError` (the API
  turns that into a clean 400).

---

## H.3 `cli.py` — the command line

```python
def main(argv=None):
    parser = argparse.ArgumentParser(prog="sentinelx", ...)
    sub = parser.add_subparsers(dest="command", required=True)
    p_run   = sub.add_parser("run",   ...)   # run pipeline, print JSON summary
    p_serve = sub.add_parser("serve", ...)   # run pipeline + start web server
    sub.add_parser("models", ...)            # list available models
    args = parser.parse_args(argv)
```

- `argparse` (stdlib) builds a command-line interface. **Subcommands** give us
  `sentinelx run`, `sentinelx serve`, `sentinelx models`. Common flags
  (`--config`, `--model`, `--seed`, `--db`) are shared.

```python
    if args.command == "models":
        from sentinelx.models import available_models
        print("\n".join(available_models())); return 0

    service = run_pipeline(config_path=args.config, overrides=_overrides(args), db_path=args.db)

    if args.command == "run":
        print(json.dumps(service.summary, indent=2)); return 0

    if args.command == "serve":
        env_port = os.environ.get("PORT") or os.environ.get("X_ZOHO_CATALYST_LISTEN_PORT")
        host = args.host or os.environ.get("HOST") or ("0.0.0.0" if env_port else cfg host)
        port = args.port or int(env_port or cfg port)
        serve(service, host=host, port=port)
        return 0
```

- `run` → run the pipeline and print the summary as pretty JSON.
- `serve` → run the pipeline, then start the web server. The **port logic** is the
  deployment glue: it reads `$PORT` (Render/Railway/Fly/Cloud Run) or
  `$X_ZOHO_CATALYST_LISTEN_PORT` (Zoho Catalyst), binds `0.0.0.0` so it's
  reachable, and falls back to the config for local runs. This one function is why
  the same code runs on your laptop *and* on any free host with no changes.

---

## Recap

Step 7 + orchestration done. `chronicle.py` writes deterministic, auditable
incident sentences. `pipeline.run_pipeline` runs the whole flow in the exact
leakage-safe order and returns a `SentinelService` that answers questions live.
`_evaluate` scores precision/recall honestly using labels only for grading.
`cli.py` exposes `run` / `serve` / `models` and makes the server deployable
anywhere via environment variables.

Next: [Code I — the API server & the dashboard](09-api.md)
