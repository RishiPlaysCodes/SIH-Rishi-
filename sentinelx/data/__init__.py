"""Data ingestion, cleaning, feature engineering and leakage-safe splitting."""

from sentinelx.data.schema import (
    EDGE_FEATURES,
    NODE_FEATURES,
    FlowRecord,
)
from sentinelx.data.synthetic import generate_synthetic_flows
from sentinelx.data.preprocess import Normalizer, clean_flows, load_cic_ids_csv
from sentinelx.data.features import build_windows, window_node_features
from sentinelx.data.splits import TemporalSplit, temporal_split, assert_no_leakage

__all__ = [
    "FlowRecord",
    "NODE_FEATURES",
    "EDGE_FEATURES",
    "generate_synthetic_flows",
    "clean_flows",
    "load_cic_ids_csv",
    "Normalizer",
    "build_windows",
    "window_node_features",
    "TemporalSplit",
    "temporal_split",
    "assert_no_leakage",
]
