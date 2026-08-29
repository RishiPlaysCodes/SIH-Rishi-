# Code I · The API Server (`api/server.py`)

Pipeline **step 8**: expose the results over HTTP (so a browser or any client can
fetch them) and serve the dashboard files. Built on stdlib `http.server` — no
web framework. (The industrial swap-in is FastAPI + uvicorn; the endpoint
contract is identical.)

> **HTTP crash course:** a browser sends a **request** to a URL with a **method**
> (`GET` = "give me data", `POST` = "here's data, do something"). The server
> replies with a **status code** (200 = OK, 400 = bad request, 404 = not found,
> 500 = server error) and a **body** (here, JSON text). A **route** is the mapping
> from a URL path to the function that handles it.

---

## I.1 The endpoints (the contract)

| Method | Path | Returns |
|--------|------|---------|
| GET | `/health` | `{"status":"ok"}` (liveness check) |
| GET | `/summary` | run metadata + detection precision/recall |
| GET | `/network/state?window=<i>` | one window's graph + per-node deviations |
| GET | `/forecast?window=<i>&k=<n>` | K-step forecast with per-step uncertainty |
| GET | `/uncertainty?window=<i>` | MC-Dropout mean/σ + label |
| GET | `/propagation` | all propagation events + Rₑ |
| GET | `/incident` | the CyberChronicle log |
| POST | `/counterfactual` | run a what-if intervention |
| POST | `/ingest` | re-run the pipeline with new parameters |
| GET | `/` , `/app.js`, `/styles.css` | the dashboard files |

---

## I.2 Serving files and JSON

```python
def build_handler(state):
    state.setdefault("ingest_lock", threading.Lock())

    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, payload, status=200):
            body = json.dumps(payload, default=_json_default, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            ...
            self.end_headers()
            self.wfile.write(body)
```

- `build_handler(state)` builds a request-handler **class** that closes over a
  `state` dict (holding the live `SentinelService` and the frontend folder). This
  pattern lets the handler access shared state cleanly.
- `_send_json` serialises a Python dict to JSON and writes the HTTP response.
- **`allow_nan=False`** is a deliberate safety net: standard JSON has no `NaN` or
  `Infinity`. If a non-finite number ever slipped through, this raises (caught as
  a 500) instead of emitting *invalid* JSON that would break strict clients.
  Defence in depth.
- `Access-Control-Allow-Origin: *` is a **CORS** header — it lets a browser page
  from any origin call this API (useful during development).

```python
        def _send_file(self, path):
            if not os.path.isfile(path):
                self._send_json({"error": "not found", ...}, 404); return
            ...
            self.send_header("Content-Type", _CONTENT_TYPES.get(ext, "application/octet-stream"))
            self.wfile.write(body)
```

- Serves a static file (the dashboard's HTML/CSS/JS) with the right content type.

---

## I.3 Routing GET requests

```python
        def do_GET(self):
            parsed = urlparse(self.path)
            route = parsed.path.rstrip("/") or "/"
            qs = parse_qs(parsed.query)
            try:
                if route == "/health":           return self._send_json({"status": "ok"})
                if route == "/summary":          return self._send_json(self.service.summary)
                if route == "/network/state":    return self._send_json(self.service.network_state(_int(qs, "window")))
                if route == "/forecast":         return self._send_json(self.service.forecast(_int(qs,"window"), _int(qs,"k")))
                if route == "/uncertainty":      return self._send_json(self.service.uncertainty(_int(qs,"window")))
                if route == "/propagation":      return self._send_json(self.service.propagation())
                if route in ("/incident","/incidents"): return self._send_json(self.service.incidents())
                if route == "/":                 return self._send_file(os.path.join(state["frontend_dir"], "index.html"))
                candidate = os.path.normpath(os.path.join(state["frontend_dir"], route.lstrip("/")))
                if candidate.startswith(state["frontend_dir"]):
                    return self._send_file(candidate)
                return self._send_json({"error": "not found"}, 404)
            except IndexError as exc:
                return self._send_json({"error": str(exc)}, 400)
            except Exception as exc:
                return self._send_json({"error": str(exc)}, 500)
```

Line by line:
- `do_GET` is called automatically for every GET request. `urlparse` splits the
  URL; `parse_qs` parses the query string (`?window=39` → `{"window": ["39"]}`).
- A simple **router**: match the path to a `SentinelService` method and return its
  result as JSON. `_int(qs, "window")` safely extracts an integer query param.
- **Static files with a security check:** for any other path we build a candidate
  file path and only serve it if it still `startswith(frontend_dir)`.
  `os.path.normpath` collapses `..` sequences, so this line **blocks path
  traversal** — an attacker can't request `/../../etc/passwd` to escape the
  frontend folder. Small line, real protection.
- **Error handling:** an `IndexError` (e.g. window out of range) becomes a clean
  **400**; any other exception becomes a **500** with a message — the server never
  crashes on a bad request.

---

## I.4 Routing POST requests (counterfactual, ingest)

```python
        def do_POST(self):
            route = urlparse(self.path).path.rstrip("/") or "/"
            try:
                body = self._read_body()               # parse JSON body (or {})
                if route == "/counterfactual":
                    return self._send_json(self.service.counterfactual(
                        action_type=body.get("action_type", "ISOLATE_NODE"),
                        window=body.get("window"), target_node=body.get("target_node"),
                        target_edge=body.get("target_edge"), port=body.get("port"),
                        rate_factor=float(body.get("rate_factor", 0.2))))
                if route == "/ingest":
                    return self._send_json(self._reingest(body))
                return self._send_json({"error": "not found"}, 404)
            except (ValueError, IndexError) as exc:
                return self._send_json({"error": str(exc)}, 400)
            except Exception as exc:
                return self._send_json({"error": str(exc)}, 500)
```

- `_read_body` reads and JSON-parses the request body (bad JSON → `ValueError` →
  400). `/counterfactual` calls the service's live what-if. Recall from Part F
  that an under-specified intervention *raises* `ValueError` — here that's caught
  and returned as a **400**, so the client gets a clear error instead of a
  misleading "no effect."

```python
        def _reingest(self, body):
            with state["ingest_lock"]:
                old_service = state["service"]
                new_service = run_pipeline(config_path=..., overrides=overrides, db_path=...)
                state["service"] = new_service
                if old_service is not None and old_service is not new_service:
                    old_service.repo.db.close()
            return {"status": "reingested", "summary": new_service.summary}
```

- `/ingest` re-runs the whole pipeline with new parameters (e.g. switch attack
  type) and swaps in the fresh `SentinelService`.
- `with state["ingest_lock"]:` — a **lock** ensures only one re-ingest happens at
  a time. Without it, two simultaneous `/ingest` calls could reset and rewrite the
  same database at once and corrupt it. It also lets us **close the old database
  connection** cleanly (no resource leak). This is a small concurrency-safety
  detail that matters for a robust server.

---

## I.5 Starting the server

```python
def serve(service, host="127.0.0.1", port=8787, verbose=False):
    state = {"service": service, "frontend_dir": os.path.abspath(_FRONTEND_DIR),
             "db_path": ..., "config_path": None, "verbose": verbose,
             "ingest_lock": threading.Lock()}
    handler = build_handler(state)
    httpd = ThreadingHTTPServer((host, port), handler)
    print(f"Sentinel-X dashboard + API listening on http://{host}:{port}")
    httpd.serve_forever()
```

- `ThreadingHTTPServer` handles each request in its own thread, so multiple
  browser requests don't block each other (that's why we needed the lock and the
  shared SQLite connection with `check_same_thread=False`).
- `serve_forever()` runs until stopped. The `host`/`port` come from the CLI's
  environment-aware logic (Part H), which is what makes it deploy anywhere.

---

## Recap

Step 8 done. A tiny, dependency-free HTTP server maps clean URLs to
`SentinelService` methods, returns JSON, and serves the dashboard files. It's
hardened where it counts: parameterised DB access, path-traversal blocking,
strict JSON (`allow_nan=False`), graceful 400/500 error handling, and a lock that
makes concurrent re-ingests safe. Same contract as FastAPI would give — just
hand-built so you can see every piece.

Next: [Code J — the dashboard frontend](10-frontend.md)
