"""Command-line entry point for Sentinel-X.

    sentinelx run       run the full pipeline and print a JSON summary
    sentinelx serve     run the pipeline then start the HTTP API + dashboard
    sentinelx models    list available world models
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", "-c", default=None, help="Path to a YAML config file")
    p.add_argument("--model", default=None, help="Override experiment.model_type")
    p.add_argument("--seed", type=int, default=None, help="Override experiment.seed")
    p.add_argument("--db", default=None, help="SQLite database path")


def _overrides(args) -> dict:
    ov: dict = {}
    exp = {}
    if getattr(args, "model", None):
        exp["model_type"] = args.model
    if getattr(args, "seed", None) is not None:
        exp["seed"] = args.seed
    if exp:
        ov["experiment"] = exp
    return ov


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sentinelx", description="Sentinel-X CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run the pipeline and print a summary")
    _add_common(p_run)

    p_serve = sub.add_parser("serve", help="Run the pipeline and start the API/dashboard")
    _add_common(p_serve)
    p_serve.add_argument("--host", default=None)
    p_serve.add_argument("--port", type=int, default=None)

    sub.add_parser("models", help="List available world models")

    args = parser.parse_args(argv)

    if args.command == "models":
        from sentinelx.models import available_models

        print("\n".join(available_models()))
        return 0

    from sentinelx.pipeline import run_pipeline

    service = run_pipeline(
        config_path=args.config, overrides=_overrides(args), db_path=args.db
    )

    if args.command == "run":
        print(json.dumps(service.summary, indent=2))
        return 0

    if args.command == "serve":
        from sentinelx.api.server import serve

        # Deployment-friendly binding: honour $PORT/$HOST (Render, Railway,
        # Fly, Cloud Run, Hugging Face Spaces, etc.) and default to 0.0.0.0 so
        # the service is reachable, falling back to the config for local runs.
        host = args.host or os.environ.get("HOST") or (
            "0.0.0.0" if os.environ.get("PORT") else service.config.get("api.host", "127.0.0.1")
        )
        port = args.port or int(os.environ.get("PORT") or service.config.get("api.port", 8787))
        serve(service, host=host, port=port)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
