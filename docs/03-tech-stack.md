# 3 · The Tech Stack (and *why* each choice)

A big theme of this project: **you don't need heavy tools to learn the ideas.**
We built the whole thing with only Python's **standard library** — the set of
modules that ship *inside* Python, so there is nothing to `pip install`.

This part explains (a) what a "standard library" even is, (b) every stdlib
module we use and why, and (c) what the "grown-up" industrial tools would be and
how they'd slot in. So you learn the concept *and* the professional ecosystem.

---

## 3.1 What "standard library" means (and why we chose it)

When you install Python, hundreds of ready-made modules come with it — for maths,
files, web servers, databases, JSON, and more. That bundle is the **standard
library** ("stdlib"). Using only stdlib means:

- **Zero install friction.** `git clone` and run. Nothing to download.
- **Runs anywhere**, including offline machines and locked-down graders.
- **Every claim is verifiable** — no hidden dependency doing the magic.
- **You see how things actually work.** Instead of calling `model.fit()` from a
  library, we *wrote* the maths, so you can read it.

The trade-off: stdlib maths is slower and simpler than industrial tools. That's
fine for learning and for a prototype. For production you'd swap in the heavy
tools (below) — and because our code hides each piece behind a clean **interface**
(a defined set of functions/classes other code talks to), the swap is a drop-in,
not a rewrite.

---

## 3.2 The stdlib modules we use

| Module | What it is | Where we use it |
|---|---|---|
| `math` | square roots, exp, etc. | distances, sigmoid, saturating functions |
| `random` | pseudo-random numbers | synthetic data, MC-Dropout, perturbations (always seeded) |
| `dataclasses` | auto-generates boilerplate for simple "record" classes | every data structure (`FlowRecord`, `GraphState`, ...) |
| `typing` / builtin generics | type hints (`list[int]`, `X | None`) | readability + tooling; hints don't affect runtime |
| `json` | read/write JSON text | API responses, storing feature vectors |
| `csv` | read comma-separated files | loading CIC-IDS2018 network data |
| `sqlite3` | a full SQL database *inside* stdlib | all persistence |
| `http.server` | a basic HTTP web server | the REST API + serving the dashboard |
| `urllib` | make HTTP requests | the test suite calls the running API |
| `hashlib` | hashing (SHA-256) | turning IPs into safe node keys; deriving seeds |
| `argparse` | parse command-line options | the `sentinelx` CLI |
| `os` / `tempfile` | files, env vars, temp dirs | deployment portability, DB fallback |
| `threading` | run code concurrently | the server serialises `/ingest` safely |
| `abc` | "abstract base class" — defines an interface | the `WorldModel` contract |
| `collections` | handy containers (`Counter`) | counting protocols/ports when building edges |

That's the entire toolbox. Notice there is **no NumPy, no PyTorch, no
FastAPI** — we re-created the *slice* of each that we needed.

---

## 3.3 Concepts we hand-built (that libraries usually give you)

This is the fun part — the things you normally import, we wrote:

- **Linear algebra** (matrix multiply, transpose, solving equations, ridge
  regression) → normally NumPy/SciPy. Ours lives in `linalg.py`, all plain
  loops. Reading it teaches you what NumPy does under the hood.
- **A YAML config reader** → normally the `PyYAML`/`ruamel.yaml` library. We wrote
  a tiny parser for the subset we need (it auto-uses a real library if one
  happens to be installed).
- **A web framework** → normally FastAPI/Flask. We used raw `http.server` with
  our own tiny router (matching URL paths to functions).
- **A dashboard framework** → normally React. We used plain HTML + vanilla
  JavaScript, plus **D3.js** loaded from a CDN in the browser (the *only* thing
  fetched from the internet, and only on the viewer's side).

---

## 3.4 The "grown-up" swap-ins (so you know the real ecosystem)

Everything above maps to a professional tool. Here is the translation table —
learn these names, they're the industry standard:

| Our stdlib piece | Industrial swap-in | What the pro tool adds |
|---|---|---|
| `linalg.py` (hand maths) | **NumPy / SciPy** | fast, vectorised array maths on huge data |
| `models/` (persistence, EWMA, ridge) | **PyTorch + PyTorch Geometric** | real neural networks, incl. **Graph Neural Networks** on GPUs |
| `graph/` (our dataclasses) | **NetworkX** (analysis) + **PyG** (learning) | rich graph algorithms + batched graph tensors |
| `data/` (csv + our features) | **pandas** (+ **Scapy** for packets) | fast dataframes; parse real PCAP capture files |
| `config.py` (our YAML) | **ruamel.yaml** + **MLflow** | full YAML + experiment tracking/versioning |
| `api/server.py` (http.server) | **FastAPI + uvicorn** | validation, async, auto docs, speed |
| `persistence/` (SQLite) | **PostgreSQL** | concurrent, production-grade database |
| `frontend/` (vanilla JS + D3) | **React + TypeScript + Vite + Tailwind** | component UI, type safety, fast builds |

### The one worth understanding deeply: Graph Neural Networks (GNNs)

Our world model is a *linear* map (ridge regression). The industrial version is
a **Graph Neural Network**. In one paragraph: a GNN lets each node update its
own feature vector by *mixing in messages from its neighbours* ("message
passing"), repeated for a few rounds, so a node's representation reflects its
*neighbourhood*, not just itself. Stack that with a **GRU** (a "Gated Recurrent
Unit" — a neural network that remembers a sequence over time) and you get a
model that learns both *graph structure* and *time dynamics*. That's the natural
upgrade path from our linear model. We deliberately started linear because it's
understandable, trains instantly, and gives an honest baseline the GNN must beat.

> All of these swap-ins are listed in `requirements-full.txt` at the repo root.
> They are documented, not required — the project runs fully without them.

---

## 3.5 Why this ordering is good engineering

Starting simple and swapping up is not a limitation — it's the *right* way to
build:

1. Prove the *idea* works with the simplest thing (baselines).
2. Measure. Only add complexity that measurably helps (ablations).
3. Keep clean interfaces so upgrades are painless.

You now know every tool in play and its professional counterpart. Next we see how
the pieces connect.

Next: [Part 4 — Architecture & data flow](04-architecture.md)
