# 5 · Project Setup (make the skeleton yourself)

Before reading the code, it helps to *create the empty skeleton* with your own
hands. This makes the folder map from Part 4 muscle-memory.

## 5.1 What a Python "package" is

A **package** is just a folder Python treats as an importable unit. The rule:
a folder becomes a package when it contains a file named `__init__.py` (it can be
empty). Sub-folders with their own `__init__.py` are **sub-packages**.

So `sentinelx/data/features.py` is imported as `sentinelx.data.features`. The
dots follow the folders.

## 5.2 Create the skeleton

From an empty directory:

```bash
mkdir -p sentinelx/{data,graph,models,forecast,analytics,persistence,narrative,api}
mkdir -p frontend tests configs docs

# make each folder a package
touch sentinelx/__init__.py
for d in data graph models forecast analytics persistence narrative api; do
  touch sentinelx/$d/__init__.py
done
```

## 5.3 The one config file that makes it a real project: `pyproject.toml`

`pyproject.toml` is the modern standard file that describes a Python project:
its name, version, Python requirement, dependencies, and command-line entry
points. Ours declares:

- `dependencies = []`  ← the proof that we need nothing external.
- an optional `[project.optional-dependencies] full = [...]` list ← the
  documented heavy swap-ins (torch, fastapi, ...), installed only if you want.
- `[project.scripts] sentinelx = "sentinelx.cli:main"` ← this line is what makes
  a `sentinelx` command exist after installation; it points at the `main`
  function in `sentinelx/cli.py`.

You don't have to install anything to run the project (we use
`python -m sentinelx.cli`), but the file documents intent and enables
`pip install -e .` if you want the short command.

## 5.4 The order we'll read the code

We follow the pipeline (Part 4), because each layer uses the previous one:

1. **Foundation** — config, seeding, linalg (used by everything).
2. **Data** — schema, synthetic, preprocess, features, splits.
3. **Graph** — types, builder.
4. **Models** — base, statistical, linear, registry.
5. **Forecast** — deviation, engine.
6. **Analytics** — the seven engines.
7. **Persistence** — schema.sql, db, repository.
8. **Pipeline glue** — narrative, pipeline, cli.
9. **API** — server.
10. **Frontend** — html, css, js.
11. **Tests** — the safety net.

Ready. Open the first walkthrough.

Next: [Code A — Foundation](code/01-foundation.md)
