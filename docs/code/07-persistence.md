# Code G · Persistence (`persistence/`)

Pipeline **step 6**: store everything the run computed, so it can be queried
later and survive as a record. Three files: `schema.sql` (the tables), `db.py`
(the connection), `repository.py` (read/write helpers).

We use **SQLite** — a complete SQL database that lives in a single file and ships
*inside* Python (`sqlite3`). No server to install. (The industrial swap-in is
PostgreSQL; the SQL is nearly identical.)

> **SQL crash course (30 seconds):** SQL ("Structured Query Language") stores data
> in **tables** (like spreadsheets). Each table has **columns** (fields) and
> **rows** (records). You `INSERT` rows, `SELECT` them back, and link tables with
> a **foreign key** (a column that points at another table's `id`). That's 90% of
> what we use.

---

## G.1 `schema.sql` — the shape of the database

The schema defines 10 tables. Here's the first, annotated:

```sql
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    dataset TEXT NOT NULL,
    model_type TEXT NOT NULL,
    config_yaml TEXT NOT NULL,
    seed INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    mlflow_run_id TEXT
);
```

- `CREATE TABLE IF NOT EXISTS` — make the table only if it doesn't already exist
  (safe to run every startup).
- `id INTEGER PRIMARY KEY AUTOINCREMENT` — a unique row number that the database
  assigns automatically. The **primary key** uniquely identifies each row.
- `TEXT NOT NULL` — a text column that must have a value.
- `config_yaml` stores the *entire config snapshot* — this is the reproducibility
  guarantee: every experiment records exactly the settings it ran with.
- `created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP` — auto-stamps the row's
  creation time.

The 10 tables and what each stores:

| Table | One row = | Key links |
|-------|-----------|-----------|
| `experiments` | one pipeline run (config + seed) | — |
| `network_snapshots` | one time-window graph | → experiment |
| `nodes` | one node in a snapshot | → snapshot |
| `edges` | one edge in a snapshot | → snapshot |
| `anomalies` | one node's deviation scores | → snapshot |
| `forecasts` | one predicted future graph (per horizon step) | → experiment, snapshot |
| `forecast_results` | uncertainty/novelty/stability for a forecast | → forecast |
| `propagation_events` | one A→B spread event | → snapshot |
| `counterfactual_runs` | one what-if simulation | → snapshot |
| `incidents` | one CyberChronicle sentence | → snapshot |

Example of a linked table (a node points back at its snapshot):

```sql
CREATE TABLE IF NOT EXISTS nodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id INTEGER REFERENCES network_snapshots(id),
    node_key TEXT NOT NULL,
    label TEXT,
    feature_vector TEXT NOT NULL,   -- the 8 numbers, stored as JSON text
    is_server INTEGER DEFAULT 0,    -- SQLite has no bool; 0/1 stands in
    status TEXT DEFAULT 'normal'
);
```

- `snapshot_id INTEGER REFERENCES network_snapshots(id)` — the **foreign key**
  linking each node to its snapshot.
- `feature_vector TEXT` — SQLite has no "list" type, so we store the vector as a
  **JSON string** like `"[5.0, 2.0, 1.0, ...]"` and parse it back when reading.
- `is_server INTEGER` — SQLite has no boolean type; we use 0/1.

At the bottom, a few **indexes**:

```sql
CREATE INDEX IF NOT EXISTS idx_nodes_snapshot ON nodes(snapshot_id);
```

- An **index** is like a book's index — it makes "find all nodes for snapshot X"
  fast instead of scanning every row. We add them on the columns we filter by.

---

## G.2 `db.py` — the connection (with a deployment-safe fallback)

```python
class Database:
    def __init__(self, path="sentinelx.db"):
        self.path, self.conn = self._connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.init_schema()
```

- `self.conn` is the live database connection.
- `row_factory = sqlite3.Row` makes query results behave like dictionaries
  (`row["name"]`) instead of bare tuples — much more readable.
- `PRAGMA foreign_keys = ON` — SQLite disables foreign-key enforcement by
  default; this turns it on so the links above are actually checked.
- `init_schema()` runs the `schema.sql` file to create the tables.

```python
    @staticmethod
    def _connect(path):
        tmp = os.path.join(tempfile.gettempdir(), os.path.basename(path) or "sentinelx.db")
        candidates = [path] if tmp == path else [path, tmp]
        for candidate in candidates:
            try:
                parent = os.path.dirname(os.path.abspath(candidate)) or "."
                os.makedirs(parent, exist_ok=True)
                conn = sqlite3.connect(candidate, check_same_thread=False)
                conn.execute("PRAGMA user_version")     # forces the file to open
                return candidate, conn
            except (sqlite3.OperationalError, OSError):
                continue
        return ":memory:", sqlite3.connect(":memory:", check_same_thread=False)
```

- **This is a real-world portability fix.** Some hosts (Zoho Catalyst, most
  serverless / read-only containers) *forbid writing files in the app folder*. If
  we tried to create `sentinelx.db` there, it would crash.
- So we try the requested path first; if that fails, fall back to the OS **temp
  directory**; if even that fails, use an **in-memory** database (`:memory:`,
  which lives only in RAM). The app keeps working everywhere with zero config.
- `check_same_thread=False` lets the multi-threaded web server share one
  connection. `try/except (sqlite3.OperationalError, OSError)` catches the
  "can't open file" errors and moves to the next candidate.

```python
    def insert(self, sql, params=()):
        cur = self.conn.execute(sql, tuple(params))
        self.conn.commit()
        return int(cur.lastrowid)

    def query(self, sql, params=()):
        return list(self.conn.execute(sql, tuple(params)).fetchall())
```

- Thin helpers. **Crucial security detail:** notice `execute(sql, params)` — the
  values are passed *separately* as `params`, never glued into the SQL string.
  This is **parameterised queries**, the defence against **SQL injection** (where
  malicious input could otherwise rewrite your query). Always do this.
- `commit()` saves changes to disk. `lastrowid` returns the id of the row just
  inserted (so callers can link child rows to it).

```python
    def reset(self):
        for t in [ ...all table names... ]:
            self.conn.execute(f"DROP TABLE IF EXISTS {t}")
        self.conn.commit()
        self.init_schema()
```

- Wipes and recreates all tables — used at the start of a fresh run so each run's
  data is clean. (The table names here are **hardcoded constants**, never user
  input, so the f-string is safe.)

---

## G.3 `repository.py` — typed read/write helpers

The **repository pattern** puts all the SQL in one place, so the rest of the code
calls friendly methods instead of writing queries. A few representative examples:

```python
class Repository:
    def __init__(self, db):
        self.db = db

    def save_experiment(self, name, dataset, model_type, config_yaml, seed, mlflow_run_id=None):
        return self.db.insert(
            "INSERT INTO experiments(name,dataset,model_type,config_yaml,seed,mlflow_run_id) "
            "VALUES(?,?,?,?,?,?)",
            (name, dataset, model_type, config_yaml, seed, mlflow_run_id))
```

- Returns the new experiment's `id`. The `?` placeholders are filled by the tuple
  — parameterised, safe.

```python
    def save_snapshot(self, experiment_id, graph):
        snapshot_id = self.db.insert("INSERT INTO network_snapshots(...) VALUES(...)", (...))
        for key in graph.node_keys():
            nd = graph.nodes[key]
            self.db.execute("INSERT INTO nodes(...) VALUES(...)",
                            (snapshot_id, nd.key, nd.label, json.dumps(nd.features),
                             int(nd.is_server), nd.status))
        for e in graph.edges:
            self.db.execute("INSERT INTO edges(...) VALUES(...)",
                            (snapshot_id, e.src, e.dst, e.protocol, e.dst_port,
                             json.dumps(e.features), e.weight))
        self.db.commit()
        return snapshot_id
```

- Saving a whole graph = one snapshot row + one row per node + one row per edge,
  all linked by `snapshot_id`. `json.dumps(nd.features)` turns the feature list
  into the JSON text the schema expects.

```python
    def get_snapshot_graph(self, snapshot_id):
        ...
        return {
            "nodes": [{"key": n["node_key"], "features": json.loads(n["feature_vector"]),
                       "status": n["status"], "is_server": bool(n["is_server"]), ...}
                      for n in nodes],
            "edges": [{"src": e["src_node_key"], "features": json.loads(e["feature_vector"]),
                       "weight": e["weight"], ...} for e in edges],
        }
```

- Reading back: `json.loads(...)` parses the stored JSON text back into a Python
  list. The result is a plain dict ready to become an API response.

The rest of the class is more of the same: `save_anomaly`, `save_forecast`,
`save_forecast_result`, `save_propagation_event`, `save_counterfactual`,
`save_incident`, and matching `get_*` readers (`get_anomalies`,
`get_incidents`, `get_propagation_events`, ...). Each is a small, named wrapper
around one INSERT or SELECT. Boring on purpose — boring persistence code is
reliable persistence code.

---

## Recap

Step 6 done. SQLite gives us a real, zero-install database. The schema mirrors the
domain (experiments → snapshots → nodes/edges/anomalies/forecasts/...). `db.py`
connects robustly (with a temp-dir fallback so it deploys anywhere) and uses
parameterised queries (no SQL injection). `repository.py` centralises all reads
and writes behind friendly methods.

Next: [Code H — narrative, pipeline, CLI](08-pipeline.md)
