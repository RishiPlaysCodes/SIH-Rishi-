"""Stdlib HTTP API serving Sentinel-X inference + the dashboard."""

from sentinelx.api.server import build_handler, serve

__all__ = ["serve", "build_handler"]
