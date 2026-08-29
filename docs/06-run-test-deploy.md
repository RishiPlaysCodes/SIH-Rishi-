# 6 · Run, Test & Deploy

You understand the whole system. Here's how to *operate* it.

---

## 6.1 Run it locally

```bash
# 1. Run the full pipeline once, print a JSON summary
python -m sentinelx.cli run

# 2. Pick a different model or seed
python -m sentinelx.cli run --model ewma --seed 7
python -m sentinelx.cli run --model persistence

# 3. Use a config file (change the scenario)
python -m sentinelx.cli run --config configs/exfiltration.yaml

# 4. List available models
python -m sentinelx.cli models
```

`python -m sentinelx.cli` means "run the `cli` module inside the `sentinelx`
package." The `run` command executes pipeline steps 1–10 and prints a summary
like:

```json
{
  "num_windows": 40, "train_windows": 28, "test_windows": 12,
  "model_type": "linear_transition",
  "detection": {"true_positives": 21, "false_positives": 0,
                "false_negatives": 0, "precision": 1.0, "recall": 1.0},
  "total_incidents": 13
}
```

## 6.2 See the dashboard

```bash
python -m sentinelx.cli serve
# → open http://127.0.0.1:8787 in your browser
```

Click through the six tabs, drag the time slider to the attack windows
(30+), watch nodes turn amber/red, and try the Counterfactual screen: isolate the
anomalous host at an early window and watch the risk drop.

## 6.3 Test it

```bash
python -m pytest            # all 74 tests, ~3s
python -m pytest -v         # verbose: list each test
python -m pytest tests/test_leakage.py    # just the leakage tests
```

Green = nothing is broken. Make this a habit after any change.

## 6.4 Deploy it for free

The app is a single dependency-free process, so it hosts almost anywhere. Full
step-by-step (with the exact clicks) lives in **[../DEPLOY.md](../DEPLOY.md)**.
The short version:

### Render (easiest — one click from GitHub)
1. Sign in to [render.com](https://render.com) **with GitHub**.
2. **New + → Blueprint** → pick your repo. It auto-reads `render.yaml`.
3. **Apply**. Open the URL it gives you.

Free tier sleeps after ~15 min idle (first request then takes ~30–60s to wake).
For a demo, hit the URL a minute beforehand to warm it up.

### Zoho Catalyst (AppSail)
```bash
npm install -g zcatalyst-cli
catalyst login
catalyst init          # choose AppSail → Python 3.11
catalyst deploy
```
The app already reads Catalyst's `X_ZOHO_CATALYST_LISTEN_PORT` and its SQLite path
auto-falls back to a temp dir (Catalyst blocks writes in the app folder), so it
just works. Note: Catalyst's free tier is *credit-limited* (AppSail bills by
uptime), unlike Render's forever-free tier.

### Docker (Hugging Face Spaces / Fly.io / Cloud Run)
A `Dockerfile` is included. `docker build -t sentinelx . && docker run -p 7860:7860 sentinelx`,
or push to any container platform.

## 6.5 How the deploy magic works (recap)

- `cli.py serve` binds `0.0.0.0` and reads the port from `$PORT` /
  `$X_ZOHO_CATALYST_LISTEN_PORT`.
- `persistence/db.py` falls back to a temp dir (then in-memory) if the app folder
  is read-only.
- The frontend uses **relative** API paths (`/summary`), so it works behind any
  hostname automatically.

Together these mean: *the same code you run locally runs on any free host, no
changes.*

Next: [Glossary](07-glossary.md) · [Build it from scratch](08-build-from-scratch.md)
