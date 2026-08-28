"""Flow cleaning, CIC-IDS2018 CSV ingestion, and feature normalisation.

Normalisation statistics are always *fit on training data only* and then applied
to test data — see :mod:`sentinelx.data.splits` and the leakage tests. Fitting a
scaler on the full dataset is one of the most common silent leaks in
network-security ML, so it is deliberately impossible to do by accident here:
``Normalizer`` must be ``fit`` before it can ``transform``.
"""

from __future__ import annotations

import csv
import hashlib
import io
import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from sentinelx.data.schema import FlowRecord
from sentinelx.linalg import Vector, clamp


# Column aliases for the common CIC-IDS2018 / CIC-flowmeter exports. Multiple
# spellings are accepted because the public CSVs are inconsistent.
_CIC_ALIASES = {
    "ts": ["Timestamp", "timestamp", "flow_start"],
    "src": ["Src IP", "Source IP", "src_ip", "Src"],
    "dst": ["Dst IP", "Destination IP", "dst_ip", "Dst"],
    "src_port": ["Src Port", "Source Port", "src_port"],
    "dst_port": ["Dst Port", "Destination Port", "dst_port"],
    "protocol": ["Protocol", "protocol"],
    "duration": ["Flow Duration", "flow_duration", "Duration"],
    "fwd_packets": ["Tot Fwd Pkts", "Total Fwd Packets", "fwd_packets"],
    "bwd_packets": ["Tot Bwd Pkts", "Total Backward Packets", "bwd_packets"],
    "fwd_bytes": ["TotLen Fwd Pkts", "Total Length of Fwd Packets", "fwd_bytes"],
    "bwd_bytes": ["TotLen Bwd Pkts", "Total Length of Bwd Packets", "bwd_bytes"],
    "label": ["Label", "label"],
}

_PROTO_NAMES = {"6": "TCP", "17": "UDP", "1": "ICMP", "0": "HOPOPT"}


def hash_asset(identifier: str, salt: str = "sentinelx") -> str:
    """Map a raw identifier (e.g. an IP) to a stable, non-reversible node key."""
    digest = hashlib.sha256(f"{salt}:{identifier}".encode()).hexdigest()
    return f"N{digest[:10]}"


def _pick(row: Dict[str, str], aliases: List[str]) -> str | None:
    for name in aliases:
        if name in row and row[name] != "":
            return row[name]
    return None


def _to_float(value: str | None, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        f = float(value)
    except (ValueError, TypeError):
        return default
    if math.isnan(f) or math.isinf(f):
        return default
    return f


def _to_int(value: str | None, default: int = 0) -> int:
    return int(_to_float(value, default))


def load_cic_ids_csv(text_or_path: str, hash_ips: bool = True, is_path: bool = True) -> List[FlowRecord]:
    """Parse a CIC-IDS2018-style CSV into :class:`FlowRecord` objects.

    ``text_or_path`` is a filesystem path when ``is_path`` (default), otherwise
    the raw CSV text (used by tests). Malformed/non-finite rows are skipped by
    :func:`clean_flows`, which callers should apply afterwards.
    """
    if is_path:
        with open(text_or_path, "r", encoding="utf-8", newline="") as fh:
            content = fh.read()
    else:
        content = text_or_path

    reader = csv.DictReader(io.StringIO(content))
    records: List[FlowRecord] = []
    for i, row in enumerate(reader):
        # Duration in CIC exports is microseconds; convert to seconds.
        raw_dur = _to_float(_pick(row, _CIC_ALIASES["duration"]), 0.0)
        duration = raw_dur / 1_000_000.0 if raw_dur > 1000 else raw_dur
        proto_raw = _pick(row, _CIC_ALIASES["protocol"]) or "TCP"
        protocol = _PROTO_NAMES.get(str(proto_raw).strip(), str(proto_raw).strip().upper())
        src = _pick(row, _CIC_ALIASES["src"]) or f"unknown-src-{i}"
        dst = _pick(row, _CIC_ALIASES["dst"]) or f"unknown-dst-{i}"
        records.append(
            FlowRecord(
                ts=_to_float(_pick(row, _CIC_ALIASES["ts"]), float(i)),
                src=hash_asset(src) if hash_ips else src,
                dst=hash_asset(dst) if hash_ips else dst,
                src_port=_to_int(_pick(row, _CIC_ALIASES["src_port"]), 0),
                dst_port=_to_int(_pick(row, _CIC_ALIASES["dst_port"]), 0),
                protocol=protocol,
                duration=duration,
                fwd_packets=_to_int(_pick(row, _CIC_ALIASES["fwd_packets"]), 0),
                bwd_packets=_to_int(_pick(row, _CIC_ALIASES["bwd_packets"]), 0),
                fwd_bytes=_to_int(_pick(row, _CIC_ALIASES["fwd_bytes"]), 0),
                bwd_bytes=_to_int(_pick(row, _CIC_ALIASES["bwd_bytes"]), 0),
                label=(_pick(row, _CIC_ALIASES["label"]) or "Benign").strip(),
            )
        )
    return records


def clean_flows(records: Sequence[FlowRecord]) -> List[FlowRecord]:
    """Drop physically impossible / empty flows and clamp non-finite fields."""
    cleaned: List[FlowRecord] = []
    for r in records:
        if r.duration < 0:
            continue
        if r.packets <= 0 and r.bytes <= 0:
            continue  # empty flow carries no signal
        if any(math.isinf(x) or math.isnan(x) for x in (r.ts, r.duration)):
            continue
        cleaned.append(r)
    cleaned.sort(key=lambda f: f.ts)
    return cleaned


@dataclass
class Normalizer:
    """Per-feature scaler that must be ``fit`` (on TRAIN data only) before use.

    Two modes:

    * ``"zscore"`` (default) — standardise to zero mean / unit variance using the
      training statistics, **without clamping**. This is deliberate and
      important: an attack drives features far outside their benign range
      (e.g. a host suddenly contacting 20 destinations instead of 2). Min-max
      scaling would clamp those spikes to 1.0 — the same value as the benign
      maximum — erasing the very signal we need. Z-scoring instead *amplifies*
      out-of-distribution values, so the world model's forecast error explodes
      on anomalous behaviour.
    * ``"minmax"`` — bounded [0, 1] scaling, kept for components (like edge
      weights) where a bounded range is preferred.
    """

    mode: str = "zscore"
    mins: List[float] = field(default_factory=list)
    maxs: List[float] = field(default_factory=list)
    means: List[float] = field(default_factory=list)
    stds: List[float] = field(default_factory=list)
    _fitted: bool = False

    def fit(self, rows: Sequence[Sequence[float]]) -> "Normalizer":
        if not rows:
            raise ValueError("Normalizer.fit received no rows")
        dim = len(rows[0])
        self.mins = [math.inf] * dim
        self.maxs = [-math.inf] * dim
        self.means = [0.0] * dim
        for row in rows:
            if len(row) != dim:
                raise ValueError("Inconsistent feature dimensionality in Normalizer.fit")
            for j, v in enumerate(row):
                self.mins[j] = min(self.mins[j], v)
                self.maxs[j] = max(self.maxs[j], v)
                self.means[j] += v
        self.means = [m / len(rows) for m in self.means]
        var = [0.0] * dim
        for row in rows:
            for j, v in enumerate(row):
                var[j] += (v - self.means[j]) ** 2
        # Population std with a small floor to avoid divide-by-zero on constants.
        self.stds = [max(math.sqrt(var[j] / len(rows)), 1e-6) for j in range(dim)]
        self._fitted = True
        return self

    def transform_vector(self, vec: Sequence[float]) -> Vector:
        if not self._fitted:
            raise RuntimeError("Normalizer.transform called before fit (potential leakage)")
        if self.mode == "minmax":
            out: Vector = []
            for j, v in enumerate(vec):
                span = self.maxs[j] - self.mins[j]
                out.append(0.0 if span <= 0 else clamp((v - self.mins[j]) / span, 0.0, 1.0))
            return out
        # z-score (unclamped)
        return [(v - self.means[j]) / self.stds[j] for j, v in enumerate(vec)]

    def transform(self, rows: Sequence[Sequence[float]]) -> List[Vector]:
        return [self.transform_vector(r) for r in rows]
