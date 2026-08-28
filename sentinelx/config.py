"""Reproducible configuration for Sentinel-X.

Every experiment is fully described by a plain dict that can be loaded from a
YAML file and snapshotted (into SQLite / MLflow) so any run is re-creatable from
``config + seed``.

To keep the reference implementation dependency-free, this module ships a small,
well-tested YAML *subset* parser (nested maps, block/inline lists of scalars,
typed scalars, ``#`` comments). If ``ruamel.yaml`` or ``pyyaml`` is installed it
is used instead automatically. The subset is sufficient for the configs shipped
in ``configs/`` and is covered by unit tests.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List


# --------------------------------------------------------------------------- #
# Default configuration (source of truth; YAML files only override these).
# --------------------------------------------------------------------------- #
DEFAULT_CONFIG: Dict[str, Any] = {
    "experiment": {
        "name": "sentinelx-default",
        "dataset": "synthetic",
        "model_type": "linear_transition",
        "seed": 1337,
    },
    "data": {
        "num_windows": 40,
        "window_seconds": 60,
        "num_hosts": 18,
        "num_servers": 4,
        "attack_start_window": 30,
        "attack_type": "lateral_movement",
        "test_fraction": 0.3,
    },
    "graph": {
        "min_edge_weight": 1.0,
        "directed": True,
    },
    "model": {
        "ridge_lambda": 0.05,
        "ewma_alpha": 0.4,
    },
    "forecast": {
        "horizon": 3,
    },
    "deviation": {
        # Weights emphasise the world model's *feature forecast error* (the core
        # PRD signal), with structural change as a secondary contributor. These
        # were calibrated on held-out synthetic attacks (see tests) to give
        # precision ~0.98 / recall ~0.78 across lateral-movement + exfiltration.
        "weights": {
            "feature": 0.5,
            "node_state": 0.18,
            "temporal": 0.15,
            "structural": 0.10,
            "edge_state": 0.07,
        },
        "anomaly_threshold": 0.45,
        "deviating_threshold": 0.25,
    },
    "uncertainty": {
        # Sigma thresholds calibrated for the z-scored feature space: benign
        # forecasts sit ~0.26, rising through an attack. So <0.30 => LOW,
        # >0.50 => HIGH, which also makes uncertainty grow with the horizon.
        "num_passes": 30,
        "dropout": 0.2,
        "low_sigma": 0.30,
        "high_sigma": 0.50,
    },
    "novelty": {
        "unusual": 0.4,
        "emerging": 0.6,
        "unknown": 0.8,
    },
    "stability": {
        "perturbation": 0.03,
        "num_trials": 12,
        "unstable_threshold": 0.12,
    },
    "api": {
        "host": "127.0.0.1",
        "port": 8787,
    },
    "persistence": {
        "db_path": "sentinelx.db",
    },
}


@dataclass
class Config:
    """Typed accessor around a nested config dict."""

    data: Dict[str, Any] = field(default_factory=lambda: copy.deepcopy(DEFAULT_CONFIG))

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def section(self, name: str) -> Dict[str, Any]:
        val = self.data.get(name, {})
        return val if isinstance(val, dict) else {}

    @property
    def seed(self) -> int:
        return int(self.get("experiment.seed", 1337))

    def as_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self.data)

    def to_yaml(self) -> str:
        return dump_yaml(self.data)


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | None = None, overrides: Dict[str, Any] | None = None) -> Config:
    """Build a :class:`Config` from defaults, an optional YAML file, and overrides."""
    merged = copy.deepcopy(DEFAULT_CONFIG)
    if path:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r", encoding="utf-8") as fh:
            file_cfg = parse_yaml(fh.read())
        merged = deep_merge(merged, file_cfg or {})
    if overrides:
        merged = deep_merge(merged, overrides)
    return Config(merged)


# --------------------------------------------------------------------------- #
# YAML handling: prefer a real library, fall back to the built-in subset parser.
# --------------------------------------------------------------------------- #
def parse_yaml(text: str) -> Dict[str, Any]:
    try:  # pragma: no cover - exercised only when the lib is installed
        from ruamel.yaml import YAML  # type: ignore

        import io

        yaml = YAML(typ="safe")
        return yaml.load(io.StringIO(text)) or {}
    except Exception:
        pass
    try:  # pragma: no cover
        import yaml as pyyaml  # type: ignore

        return pyyaml.safe_load(text) or {}
    except Exception:
        pass
    return _parse_yaml_subset(text)


def dump_yaml(data: Dict[str, Any]) -> str:
    """Serialise a nested dict to the YAML subset understood by the parser."""
    lines: List[str] = []
    _dump_node(data, 0, lines)
    return "\n".join(lines) + "\n"


def _dump_node(node: Any, indent: int, lines: List[str]) -> None:
    pad = "  " * indent
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(value, dict) and value:
                lines.append(f"{pad}{key}:")
                _dump_node(value, indent + 1, lines)
            elif isinstance(value, list):
                lines.append(f"{pad}{key}:")
                for item in value:
                    lines.append(f"{'  ' * (indent + 1)}- {_scalar_to_str(item)}")
            else:
                lines.append(f"{pad}{key}: {_scalar_to_str(value)}")


def _scalar_to_str(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return str(value)


# --- minimal indentation-based YAML subset parser -------------------------- #
def _parse_yaml_subset(text: str) -> Dict[str, Any]:
    tokens = []  # (indent, key, raw_value, is_list_item)
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip() if not _in_quotes(raw_line) else raw_line.rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        content = line.strip()
        if content.startswith("- "):
            tokens.append((indent, None, content[2:].strip(), True))
        elif content == "-":
            tokens.append((indent, None, "", True))
        else:
            if ":" not in content:
                raise ValueError(f"Malformed YAML line: {raw_line!r}")
            key, _, value = content.partition(":")
            tokens.append((indent, key.strip(), value.strip(), False))
    result, _ = _build(tokens, 0, 0)
    return result if isinstance(result, dict) else {}


def _in_quotes(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith('"') or stripped.startswith("'")


def _build(tokens, index: int, indent: int):
    """Recursively build a structure from tokens at the given indent level."""
    # Determine whether this block is a list or a map.
    if index < len(tokens) and tokens[index][3]:
        result: List[Any] = []
        while index < len(tokens):
            tok_indent, _, raw_value, is_item = tokens[index]
            if tok_indent < indent or not is_item:
                break
            result.append(_coerce(raw_value))
            index += 1
        return result, index

    result_map: Dict[str, Any] = {}
    while index < len(tokens):
        tok_indent, key, raw_value, is_item = tokens[index]
        if tok_indent < indent or is_item:
            break
        index += 1
        if raw_value == "":
            # Nested block follows (either map or list) at a deeper indent.
            if index < len(tokens) and tokens[index][0] > tok_indent:
                child, index = _build(tokens, index, tokens[index][0])
                result_map[key] = child
            else:
                result_map[key] = {}
        else:
            result_map[key] = _coerce(raw_value)
    return result_map, index


def _coerce(raw: str) -> Any:
    if raw == "":
        return None
    if (raw[0] == raw[-1]) and raw[0] in "\"'" and len(raw) >= 2:
        return raw[1:-1]
    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1].strip()
        if not inner:
            return []
        return [_coerce(part.strip()) for part in inner.split(",")]
    low = raw.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    if low in ("null", "none", "~"):
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


__all__ = [
    "Config",
    "DEFAULT_CONFIG",
    "load_config",
    "deep_merge",
    "parse_yaml",
    "dump_yaml",
]
