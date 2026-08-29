# Code K · The Tests (`tests/`)

The **test suite** is the safety net: automated checks that fail loudly if any
change breaks behaviour. There are **74 tests**; they run in ~3 seconds with only
`pytest` installed. Run them any time with:

```bash
python -m pytest
```

> **What is pytest?** A tool that finds every function named `test_*`, runs it,
> and reports pass/fail. Inside a test you `assert` something is true; if it
> isn't, the test fails with a clear message. That's the whole idea.

---

## K.1 `conftest.py` — shared fixtures

```python
@pytest.fixture(scope="session")
def flows():
    return clean_flows(generate_synthetic_flows(num_windows=40, num_hosts=14,
                       num_servers=4, attack_start_window=30, seed=1337))

@pytest.fixture(scope="session")
def graph_sequence(flows):
    return build_graph_sequence(flows, 60, 1.0)

@pytest.fixture
def tmp_db(tmp_path):
    return str(tmp_path / "sentinelx_test.db")
```

- A **fixture** is reusable test setup. Any test that lists `flows` or
  `graph_sequence` as a parameter automatically receives it — pytest builds it and
  injects it. This is **dependency injection**.
- `scope="session"` means "build it once and share across all tests" (fast).
- `tmp_path` is a built-in pytest fixture giving a fresh temporary folder, so
  database tests never touch real files.

---

## K.2 What each test file protects

### `test_config.py`
Checks the config loads, overrides merge correctly, and — importantly — that our
hand-written YAML reader **round-trips** (write YAML → read it back → same data)
and correctly types values (`true`→bool, `42`→int, comments ignored).

### `test_linalg.py`
Verifies the maths from Part A: `matmul`/`transpose` give known results, `solve`
recovers the answer to a known linear system, and `ridge_fit` **recovers a known
linear map** (feed it data from `y = 2x + 3`, check it learns ≈ [2, 3]). If the
maths ever breaks, these fail first.

### `test_data.py`
Synthetic data is **deterministic** (same seed → identical flows), attacks only
appear after the attack window, cleaning drops junk, windowing has no gaps,
features have the right shape, the CIC CSV loader parses correctly and hashes IPs,
and the `Normalizer` refuses to transform before `fit` (leakage guard) while
z-score **amplifies outliers** and min-max stays bounded.

### `test_leakage.py` (the most important file)
This encodes the anti-cheating discipline from concepts. It checks:
- the temporal split is **chronological** and covers everything,
- `assert_no_leakage` **catches** overlap, "time travel", empty train, and
  unordered splits (each is a separate test that expects an error),
- the normaliser fit on **train-only** differs from one fit on **all data** — i.e.
  the attack really does shift the distribution, so leaking it *would* matter.

If someone ever reintroduces leakage, these tests go red immediately.

### `test_graph.py`
The graph builder produces the right number of snapshots, nodes/edges exist,
servers are detected, embeddings are a fixed length, `clone()` is a true deep
copy (editing the clone doesn't touch the original), and `to_json` has the right
shape.

### `test_models.py`
Every model builds from the registry, unknown model names raise, persistence
predicts the last state, `predict_sequence` returns the right length,
`apply_dropout` is deterministic under a fixed seed, and empty history raises.

### `test_forecast.py`
`saturate` stays bounded, deviation is ~0 when prediction equals reality,
rolling deviation covers the timeline, **attack windows score higher than benign**,
and a full **detection precision ≥ 0.9 / recall ≥ 0.6** on the synthetic attack.
This is the test that proves the whole idea works.

### `test_analytics.py`
Propagation detects the chain (Rₑ ≥ 0, source ≠ target), uncertainty grows in the
attack context and needs ≥ 2 passes, novelty rates attack graphs more novel than
benign, stability is bounded, an **early** ISOLATE reduces risk (ΔRisk > 0),
`apply_intervention` actually removes the node, invalid/under-specified
interventions **raise** (the loophole fix), and MITRE/explain map correctly.

### `test_persistence.py`
The schema creates all 10 tables, experiments/snapshots/anomalies/incidents
round-trip through SQLite, and `reset()` clears data.

### `test_pipeline_api.py`
The end-to-end check: the pipeline summary hits the quality bar, the run is
**reproducible** (two runs give identical detection), the service answers
`network_state`/`forecast`/`counterfactual`, out-of-range windows raise, and —
by starting the real server on a port — **every HTTP endpoint returns 200**, a
counterfactual with a target works, and an under-specified one returns **400**.

---

## K.3 Why this matters

Tests are not busywork. They are how you *change code with confidence*: make an
edit, run `pytest`, and if it's green you didn't break anything. The leakage and
detection tests in particular encode the project's integrity — they make it
impossible to accidentally ship a version that cheats or stops detecting attacks.

That completes the code tour. You have now seen every file, browser to database.

Back to: [documentation index](../README.md) · Next practical steps:
[run / test / deploy](../06-run-test-deploy.md).
