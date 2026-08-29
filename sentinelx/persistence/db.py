"""Thin SQLite connection wrapper.

Uses only the standard-library ``sqlite3`` module. Postgres is the documented
production swap-in (see requirements-full.txt) but is intentionally out of scope
for the prototype per the PRD.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
from collections.abc import Iterable
from typing import Any

_SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema.sql")


class Database:
    def __init__(self, path: str = "sentinelx.db"):
        # check_same_thread=False so the stdlib HTTP server can share it.
        self.path, self.conn = self._connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.init_schema()

    @staticmethod
    def _connect(path: str):
        """Open the SQLite file, falling back when the target dir isn't writable.

        Some hosts (Zoho Catalyst AppSail, and most serverless / read-only
        container filesystems) forbid writing in the app directory. Rather than
        crash, we transparently fall back to the OS temp dir, then to an
        in-memory database as a last resort — so deployment friction stays at
        zero on every platform.
        """
        tmp = os.path.join(tempfile.gettempdir(), os.path.basename(path) or "sentinelx.db")
        candidates = [path] if tmp == path else [path, tmp]
        for candidate in candidates:
            try:
                parent = os.path.dirname(os.path.abspath(candidate)) or "."
                os.makedirs(parent, exist_ok=True)
                conn = sqlite3.connect(candidate, check_same_thread=False)
                conn.execute("PRAGMA user_version")  # forces the file to open
                return candidate, conn
            except (sqlite3.OperationalError, OSError):
                continue
        # Last resort: ephemeral in-memory DB (the app still serves fine).
        return ":memory:", sqlite3.connect(":memory:", check_same_thread=False)

    def init_schema(self) -> None:
        with open(_SCHEMA_PATH, "r", encoding="utf-8") as fh:
            self.conn.executescript(fh.read())
        self.conn.commit()

    def execute(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        cur = self.conn.execute(sql, tuple(params))
        return cur

    def insert(self, sql: str, params: Iterable[Any] = ()) -> int:
        cur = self.conn.execute(sql, tuple(params))
        self.conn.commit()
        return int(cur.lastrowid)

    def query(self, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        return list(self.conn.execute(sql, tuple(params)).fetchall())

    def query_one(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Row | None:
        row = self.conn.execute(sql, tuple(params)).fetchone()
        return row

    def commit(self) -> None:
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def reset(self) -> None:
        """Drop and recreate all tables (used for fresh pipeline runs / tests)."""
        tables = [
            "forecast_results", "forecasts", "anomalies", "propagation_events",
            "counterfactual_runs", "incidents", "edges", "nodes",
            "network_snapshots", "experiments",
        ]
        for t in tables:
            self.conn.execute(f"DROP TABLE IF EXISTS {t}")
        self.conn.commit()
        self.init_schema()
