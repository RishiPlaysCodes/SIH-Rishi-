"""Standard-library HTTP API (PRD §2.16) + static dashboard host.

Implemented on ``http.server`` so it runs with zero dependencies. FastAPI +
uvicorn are the documented production swap-in (requirements-full.txt): the
endpoint contract below is identical, so the React dashboard and any API client
port over unchanged.

Endpoints
    GET  /health
    GET  /summary
    GET  /network/state?window=<i>
    GET  /forecast?window=<i>&k=<n>
    GET  /uncertainty?window=<i>
    GET  /propagation
    GET  /incident            (alias: /incidents)
    POST /counterfactual      body: {action_type,window,target_node,target_edge,port,rate_factor}
    POST /ingest              body: {attack_type,seed,model_type,...}  (re-runs the pipeline)
    GET  /                    dashboard (frontend/index.html)
    GET  /<static>            dashboard assets (app.js, styles.css)
"""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict
from urllib.parse import parse_qs, urlparse

_FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "frontend")

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
}


def build_handler(state: Dict[str, Any]):
    """Create a request handler class bound to a mutable ``state`` dict.

    ``state['service']`` is the live :class:`SentinelService`; ``/ingest`` may
    replace it with a freshly-run pipeline.
    """

    class Handler(BaseHTTPRequestHandler):
        server_version = "SentinelX/0.1"

        # -- helpers -------------------------------------------------------- #
        def _send_json(self, payload: Any, status: int = 200) -> None:
            body = json.dumps(payload, default=_json_default).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.end_headers()
            self.wfile.write(body)

        def _send_file(self, path: str) -> None:
            if not os.path.isfile(path):
                self._send_json({"error": "not found", "path": os.path.basename(path)}, 404)
                return
            ext = os.path.splitext(path)[1]
            with open(path, "rb") as fh:
                body = fh.read()
            self.send_response(200)
            self.send_header("Content-Type", _CONTENT_TYPES.get(ext, "application/octet-stream"))
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def _read_body(self) -> Dict[str, Any]:
            length = int(self.headers.get("Content-Length", 0) or 0)
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            if not raw:
                return {}
            try:
                return json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON body: {exc}") from exc

        @property
        def service(self):
            return state["service"]

        def log_message(self, fmt, *args):  # silence default noisy logging
            if state.get("verbose"):
                super().log_message(fmt, *args)

        # -- routing -------------------------------------------------------- #
        def do_OPTIONS(self):  # noqa: N802 - CORS preflight
            self._send_json({}, 204)

        def do_GET(self):  # noqa: N802
            parsed = urlparse(self.path)
            route = parsed.path.rstrip("/") or "/"
            qs = parse_qs(parsed.query)
            try:
                if route == "/health":
                    return self._send_json({"status": "ok"})
                if route == "/summary":
                    return self._send_json(self.service.summary)
                if route == "/network/state":
                    return self._send_json(self.service.network_state(_int(qs, "window")))
                if route == "/forecast":
                    return self._send_json(
                        self.service.forecast(_int(qs, "window"), _int(qs, "k"))
                    )
                if route == "/uncertainty":
                    return self._send_json(self.service.uncertainty(_int(qs, "window")))
                if route == "/propagation":
                    return self._send_json(self.service.propagation())
                if route in ("/incident", "/incidents"):
                    return self._send_json(self.service.incidents())
                if route == "/":
                    return self._send_file(os.path.join(state["frontend_dir"], "index.html"))
                # static asset
                candidate = os.path.normpath(
                    os.path.join(state["frontend_dir"], route.lstrip("/"))
                )
                if candidate.startswith(state["frontend_dir"]):
                    return self._send_file(candidate)
                return self._send_json({"error": "not found"}, 404)
            except IndexError as exc:
                return self._send_json({"error": str(exc)}, 400)
            except Exception as exc:  # pragma: no cover - defensive
                return self._send_json({"error": str(exc)}, 500)

        def do_POST(self):  # noqa: N802
            parsed = urlparse(self.path)
            route = parsed.path.rstrip("/") or "/"
            try:
                body = self._read_body()
                if route == "/counterfactual":
                    return self._send_json(
                        self.service.counterfactual(
                            action_type=body.get("action_type", "ISOLATE_NODE"),
                            window=body.get("window"),
                            target_node=body.get("target_node"),
                            target_edge=body.get("target_edge"),
                            port=body.get("port"),
                            rate_factor=float(body.get("rate_factor", 0.2)),
                        )
                    )
                if route == "/ingest":
                    return self._send_json(self._reingest(body))
                return self._send_json({"error": "not found"}, 404)
            except (ValueError, IndexError) as exc:
                return self._send_json({"error": str(exc)}, 400)
            except Exception as exc:  # pragma: no cover - defensive
                return self._send_json({"error": str(exc)}, 500)

        def _reingest(self, body: Dict[str, Any]) -> Dict[str, Any]:
            from sentinelx.pipeline import run_pipeline

            overrides: Dict[str, Any] = {}
            exp = {}
            if "seed" in body:
                exp["seed"] = int(body["seed"])
            if "model_type" in body:
                exp["model_type"] = str(body["model_type"])
            if exp:
                overrides["experiment"] = exp
            data = {}
            for key in ("attack_type", "num_hosts", "num_servers", "attack_start_window", "num_windows"):
                if key in body:
                    data[key] = body[key]
            if data:
                overrides["data"] = data
            new_service = run_pipeline(
                config_path=state.get("config_path"),
                overrides=overrides,
                db_path=state.get("db_path"),
            )
            state["service"] = new_service
            return {"status": "reingested", "summary": new_service.summary}

    return Handler


def _int(qs: Dict[str, Any], key: str):
    if key in qs and qs[key]:
        return int(qs[key][0])
    return None


def _json_default(obj):
    if isinstance(obj, set):
        return sorted(obj)
    return str(obj)


def serve(service, host: str = "127.0.0.1", port: int = 8787, verbose: bool = False) -> None:
    state: Dict[str, Any] = {
        "service": service,
        "frontend_dir": os.path.abspath(_FRONTEND_DIR),
        "db_path": getattr(getattr(service, "repo", None), "db", None)
        and service.repo.db.path,
        "config_path": None,
        "verbose": verbose,
    }
    handler = build_handler(state)
    httpd = ThreadingHTTPServer((host, port), handler)
    print(f"Sentinel-X dashboard + API listening on http://{host}:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover
        pass
    finally:
        httpd.server_close()
