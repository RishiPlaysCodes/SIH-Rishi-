"""Thin SQLite connection wrapper.

Uses only the standard-library ``sqlite3`` module. Postgres is the documented
production swap-in (see requirements-full.txt) but is intentionally out of scope
for the prototype per the PRD.
"""

from __future__ import annotations

import os
import sqlite3
from typing import Any, Iterable, List, Optional

_SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema.sql")


class Database:
    def __init__(self, path: str = "sentinelx.db"):
        self.path = path
        # check_same_thread=False so the stdlib HTTP server can share it.
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.init_schema()

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

    def query(self, sql: str, params: Iterable[Any] = ()) -> List[sqlite3.Row]:
        return list(self.conn.execute(sql, tuple(params)).fetchall())

    def query_one(self, sql: str, params: Iterable[Any] = ()) -> Optional[sqlite3.Row]:
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
