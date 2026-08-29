# Sentinel-X — The Complete Learning Documentation

Welcome. This is not a normal API doc. It is a **teach-from-zero handbook**.

If you have never heard of graph machine learning, forecasting, or anomaly
detection — that is fine. Read these pages **in order**, and by the end you will
(a) understand every concept the project uses, (b) understand every line of
code, and (c) be able to rebuild the whole thing yourself.

> **Golden rule of this book:** we never use a word before we explain it. Every
> technical term is defined the first time it appears, in plain language, with a
> real-world analogy.

---

## How to read this (learning path)

Read top to bottom. Each part assumes you read the previous one.

| # | File | What you learn | Needs code? |
|---|------|----------------|-------------|
| 0 | [00-START-HERE.md](00-START-HERE.md) | How to use this book, how to set up your machine | No |
| 1 | [01-what-is-sentinelx.md](01-what-is-sentinelx.md) | The problem, the big idea, the intuition | No |
| 2 | [02-concepts.md](02-concepts.md) | Every concept from scratch (graphs → forecasting → uncertainty) | No |
| 3 | [03-tech-stack.md](03-tech-stack.md) | Every technology used and *why* | A little |
| 4 | [04-architecture.md](04-architecture.md) | How all the pieces connect; the data flow | A little |
| 5 | [05-project-setup.md](05-project-setup.md) | Folder layout; create the skeleton yourself | Yes |
| 6 | **Code walkthroughs** (below) | Every file, explained chunk by chunk | Yes |
| 7 | [06-run-test-deploy.md](06-run-test-deploy.md) | Run it, test it, deploy it free | Yes |
| 8 | [07-glossary.md](07-glossary.md) | Quick dictionary of every term | — |
| 9 | [08-build-from-scratch.md](08-build-from-scratch.md) | Rebuild the project step by step | Yes |

### Code walkthroughs (Part 6, read in this order)

| File | Covers |
|------|--------|
| [code/01-foundation.md](code/01-foundation.md) | `__init__.py`, `config.py`, `seeding.py`, `linalg.py` |
| [code/02-data.md](code/02-data.md) | `data/schema.py`, `synthetic.py`, `preprocess.py`, `features.py`, `splits.py` |
| [code/03-graph.md](code/03-graph.md) | `graph/types.py`, `graph/builder.py` |
| [code/04-models.md](code/04-models.md) | `models/base.py`, `statistical.py`, `linear.py`, `registry.py` |
| [code/05-forecast.md](code/05-forecast.md) | `forecast/deviation.py`, `forecast/engine.py` |
| [code/06-analytics.md](code/06-analytics.md) | `analytics/*` (propagation, uncertainty, novelty, stability, counterfactual, mitre, explain) |
| [code/07-persistence.md](code/07-persistence.md) | `persistence/schema.sql`, `db.py`, `repository.py` |
| [code/08-pipeline.md](code/08-pipeline.md) | `narrative/chronicle.py`, `pipeline.py`, `cli.py` |
| [code/09-api.md](code/09-api.md) | `api/server.py` |
| [code/10-frontend.md](code/10-frontend.md) | `frontend/index.html`, `styles.css`, `app.js` |
| [code/11-tests.md](code/11-tests.md) | Every test file and what it protects |

---

## What is this project, in one sentence?

> Sentinel-X watches a computer network as a **graph that changes over time**,
> **learns how it normally behaves**, and **predicts where it is heading** — so
> it can warn you about an attack *before* it finishes, and even let you
> simulate "what if I block this host?" before you touch the real network.

If that sounds like a lot — don't worry. Part 1 breaks it down with pictures and
analogies before any math or code.

---

## A note on honesty (important)

This project was built to run **with zero external libraries** (only what ships
with Python). That was a deliberate choice so it runs anywhere and every claim
is verifiable. Wherever a "real-world heavy" tool (PyTorch, FastAPI, React)
would normally be used, we explain *both*: what we did, and what the
industrial-strength swap-in looks like. So you learn the concept **and** the
professional tooling around it.

Turn to [00-START-HERE.md](00-START-HERE.md) to begin.
