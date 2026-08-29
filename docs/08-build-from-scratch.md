# 8 · Build It From Scratch

The final test of understanding: rebuild Sentinel-X yourself, from an empty
folder. This chapter is a **guided path**, not new code — at each step it tells
you *what* to build, *why*, and *which walkthrough* to re-read. Build in this
order because each layer uses the previous one, and **run something after every
step** so you always have a working (if small) program.

> Tip: keep the real repo open in a second window. Try to write each file
> yourself first, then compare. Copying teaches nothing; struggling then checking
> teaches everything.

---

## Milestone 0 — Skeleton (10 min)
Create the folders and `__init__.py` files (see
[Part 5](05-project-setup.md)). Write `pyproject.toml` with `dependencies = []`.
**Check:** `python -c "import sentinelx"` runs without error.

## Milestone 1 — Foundation ([Code A](code/01-foundation.md))
Write `seeding.py` (the `Rng` class) and `linalg.py` (vectors, `matmul`,
`transpose`, `solve`, `ridge_fit`). Then `config.py` (`DEFAULT_CONFIG`, `Config`,
`load_config`).
**Check:** in a REPL, `ridge_fit` recovers `y = 2x + 3` (≈ weights [2, 3]); a
`Config` reads `cfg.seed`. *Write these two tests now* — they'll catch mistakes
for the rest of the build.

## Milestone 2 — Data ([Code B](code/02-data.md))
Write `schema.py` (`FlowRecord`, `NODE_FEATURES`, `EDGE_FEATURES`), then
`synthetic.py` (profile-based generator with a compromise chain), then
`features.py` (`build_windows` with **grid anchoring**, `window_node_features`),
then `preprocess.py` (`Normalizer` with the fit-before-transform guard;
`load_cic_ids_csv` can come later), then `splits.py` (`temporal_split`,
`assert_no_leakage`).
**Check:** generate flows, build windows, print how many; confirm attacks only
appear after window 30. Write the leakage tests now — they're the soul of the
project.

## Milestone 3 — Graph ([Code C](code/03-graph.md))
Write `graph/types.py` (`NodeState`, `EdgeState`, `GraphState` with
`feature_matrix`, `edge_set`, `adjacency`, `embedding`, `clone`) and
`graph/builder.py` (`build_graph_state`, `build_graph_sequence`, server
detection).
**Check:** build a graph sequence from synthetic flows; a snapshot has the right
node/edge counts and detects `SERVER-*`.

## Milestone 4 — Models ([Code D](code/04-models.md))
Write `models/base.py` (the `WorldModel` ABC + `apply_dropout` +
`predict_sequence`), `statistical.py` (Persistence, EWMA), `linear.py`
(`LinearTransitionModel` using your `ridge_fit`), `registry.py`.
**Check:** fit the linear model on a few windows and call `predict_next`; it
returns a graph with predicted features.

## Milestone 5 — Forecast & deviation ([Code E](code/05-forecast.md))
Write `forecast/deviation.py` (`saturate`, the five error components,
`compute_deviation`) and `forecast/engine.py` (`ForecastEngine` with
`rolling_deviation`, `score_window`, `apply_statuses`).
**Check (the big one):** normalise with train-only stats, fit the model,
run `rolling_deviation`, and confirm the attacker scores far higher than benign
hosts. This is the moment the detector *works*.

## Milestone 6 — Analytics ([Code F](code/06-analytics.md))
Add the seven engines one at a time, each with a quick check:
propagation (finds the chain), uncertainty (grows in attack windows), novelty
(attack > benign), stability (bounded), counterfactual (early ISOLATE drops
risk), mitre + explain (map correctly).

## Milestone 7 — Persistence ([Code G](code/07-persistence.md))
Write `schema.sql`, `persistence/db.py` (with the temp-dir fallback),
`repository.py`.
**Check:** save an experiment + a snapshot, read them back; `reset()` clears.

## Milestone 8 — Narrative + Pipeline + CLI ([Code H](code/08-pipeline.md))
Write `narrative/chronicle.py`, then `pipeline.py` (`run_pipeline`,
`SentinelService`, `_persist_run`, `_evaluate`), then `cli.py`.
**Check:** `python -m sentinelx.cli run` prints a summary with precision/recall.
🎉 You now have the whole brain working end to end.

## Milestone 9 — API + Frontend ([Code I](code/09-api.md), [Code J](code/10-frontend.md))
Write `api/server.py` (router + JSON + static files + the safety details), then
the `frontend/` files (HTML structure, CSS theme, `app.js` with `fetch` +
`drawGraph`).
**Check:** `python -m sentinelx.cli serve`, open the browser, see the live graph.

## Milestone 10 — Tests & Deploy ([Code K](code/11-tests.md), [Part 6](06-run-test-deploy.md))
Fill out the test suite until `python -m pytest` is green. Add `render.yaml` /
`Dockerfile`. Deploy free and open the public URL.

---

## What you'll have learned
By finishing, you'll have hands-on command of: dynamic graphs, feature
engineering, z-score normalisation, ridge regression (and the linear algebra
under it), forecasting and residual-based anomaly detection, MC-Dropout
uncertainty, graph propagation / epidemiology metrics, OOD/novelty, counterfactual
reasoning, a REST API, a D3 dashboard, SQL persistence, testing discipline, and
leakage-safe ML practice — the exact toolkit behind modern ML systems, built with
your own hands.

That is true mastery: not "I ran it," but "I could build it again."

Back to the [index](README.md).
