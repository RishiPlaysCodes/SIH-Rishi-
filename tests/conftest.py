"""Shared pytest fixtures for the Sentinel-X test suite."""

import os
import sys

import pytest

# Ensure the package is importable when tests run from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentinelx.data import clean_flows, generate_synthetic_flows  # noqa: E402
from sentinelx.graph import build_graph_sequence  # noqa: E402


@pytest.fixture(scope="session")
def flows():
    return clean_flows(
        generate_synthetic_flows(
            num_windows=40, num_hosts=14, num_servers=4, attack_start_window=30, seed=1337
        )
    )


@pytest.fixture(scope="session")
def graph_sequence(flows):
    return build_graph_sequence(flows, 60, 1.0)


@pytest.fixture
def tmp_db(tmp_path):
    return str(tmp_path / "sentinelx_test.db")
