# Deploying Sentinel-X for free

Sentinel-X is a single, dependency-free Python service that serves both the REST
API and the dashboard on one port. That makes it trivial to host for free — no
build step, no database server, nothing to install.

Pick **one** of the options below. All of them give you a public URL you can open
in a browser, so you never have to run it locally again.

---

## Option A — Render (easiest, connects straight to GitHub) ✅ recommended

Free "Web Service", no credit card required.

1. Go to <https://render.com> and sign in **with GitHub**.
2. Click **New +  →  Blueprint**.
3. Select this repository (`SIH-Rishi-`). Render detects `render.yaml`
   automatically.
4. Click **Apply**. Render builds and starts it (about a minute).
5. Open the URL it gives you, e.g. `https://sentinel-x.onrender.com`.

That's it. The dashboard loads at `/`, the API at `/summary`, `/network/state`, etc.

> The free tier sleeps after ~15 minutes of inactivity; the first request after
> that takes ~30–60 s to wake up, then it's instant again.

If you don't want to use the Blueprint: **New + → Web Service → pick the repo**,
then set **Build Command** = `python -m compileall sentinelx` and **Start
Command** = `python -m sentinelx.cli serve`. Leave everything else default.

---

## Option B — Hugging Face Spaces (free forever, no credit card)

Uses the included `Dockerfile`.

1. Go to <https://huggingface.co/spaces> → **Create new Space**.
2. Choose **Docker → Blank**, visibility **Public**, hardware **CPU basic (free)**.
3. In the new Space, either connect this GitHub repo or push the files to the
   Space's git remote. The Space builds the `Dockerfile` and runs on port 7860.
4. Open the Space URL — the dashboard is live.

---

## Option C — Railway / Fly.io / Google Cloud Run

- **Railway**: New Project → Deploy from GitHub repo. It uses the `Procfile`.
- **Fly.io / Cloud Run**: use the `Dockerfile` (`fly launch` / `gcloud run deploy
  --source .`). The server automatically binds `0.0.0.0` on the platform's
  `$PORT`.

---

## Run it locally (optional)

```bash
python -m sentinelx.cli serve
# → open http://127.0.0.1:8787
```

---

## Verify a deployment

Once it's live, these should all return `200`:

```bash
curl https://YOUR-URL/health
curl https://YOUR-URL/summary
curl "https://YOUR-URL/forecast?window=39&k=3"
```

And opening `https://YOUR-URL/` in a browser shows the dashboard with the
Network / Forecast / Uncertainty / Propagation / Counterfactual / CyberChronicle
tabs.

## Configuration knobs (optional env vars)

| Variable | Effect |
|---|---|
| `PORT` | Port to bind (set automatically by most hosts) |
| `HOST` | Bind address (defaults to `0.0.0.0` when `PORT` is set) |

The scenario/model can be changed by editing `configs/default.yaml` (or via the
dashboard's re-ingest and the `POST /ingest` endpoint) — see `README.md`.
