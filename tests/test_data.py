"""Data layer: synthetic generation, CIC parsing, features, normalisation."""

import pytest

from sentinelx.data import (
    NODE_FEATURES,
    Normalizer,
    clean_flows,
    generate_synthetic_flows,
    load_cic_ids_csv,
)
from sentinelx.data.features import build_windows, window_node_features


def test_synthetic_is_deterministic():
    a = generate_synthetic_flows(num_windows=10, seed=42)
    b = generate_synthetic_flows(num_windows=10, seed=42)
    assert len(a) == len(b)
    assert [f.ts for f in a] == [f.ts for f in b]


def test_synthetic_has_attack_labels():
    flows = generate_synthetic_flows(num_windows=40, attack_start_window=30, seed=1)
    assert any(f.is_attack for f in flows)
    # attacks only appear from the attack window onward (grid-aligned windows)
    t_min = (min(f.ts for f in flows) // 60) * 60
    for f in flows:
        if f.is_attack:
            assert (f.ts - t_min) // 60 >= 30


def test_clean_flows_drops_empty_and_negative():
    flows = generate_synthetic_flows(num_windows=5, seed=1)
    from sentinelx.data.schema import FlowRecord

    flows.append(FlowRecord(ts=1.0, src="a", dst="b", src_port=1, dst_port=2,
                            protocol="TCP", duration=-1.0, fwd_packets=1, bwd_packets=1,
                            fwd_bytes=1, bwd_bytes=1))
    flows.append(FlowRecord(ts=2.0, src="a", dst="b", src_port=1, dst_port=2,
                            protocol="TCP", duration=1.0, fwd_packets=0, bwd_packets=0,
                            fwd_bytes=0, bwd_bytes=0))
    cleaned = clean_flows(flows)
    assert all(f.duration >= 0 and f.packets + f.bytes > 0 for f in cleaned)


def test_build_windows_no_gaps():
    flows = clean_flows(generate_synthetic_flows(num_windows=12, seed=3))
    windows = build_windows(flows, 60)
    indices = [w.index for w in windows]
    assert indices == list(range(len(windows)))  # contiguous, no gaps


def test_node_features_shape_and_values():
    flows = clean_flows(generate_synthetic_flows(num_windows=6, seed=3))
    windows = build_windows(flows, 60)
    feats = window_node_features(windows[0].flows)
    assert feats
    for vec in feats.values():
        assert len(vec) == len(NODE_FEATURES)
        assert all(isinstance(v, float) for v in vec)


def test_load_cic_ids_csv_from_text():
    csv_text = (
        "Timestamp,Src IP,Dst IP,Src Port,Dst Port,Protocol,Flow Duration,"
        "Tot Fwd Pkts,Tot Bwd Pkts,TotLen Fwd Pkts,TotLen Bwd Pkts,Label\n"
        "1000,10.0.0.1,10.0.0.2,50000,443,6,2000000,10,12,1000,2000,Benign\n"
        "1001,10.0.0.1,10.0.0.3,50001,80,6,500000,5,0,400,0,DDoS\n"
    )
    records = load_cic_ids_csv(csv_text, is_path=False)
    assert len(records) == 2
    assert records[0].protocol == "TCP"
    assert records[0].duration == 2.0  # microseconds -> seconds
    assert records[1].failed  # bwd_packets == 0
    assert records[1].is_attack
    # IPs are hashed, never stored raw
    assert not records[0].src.count(".")


def test_normalizer_requires_fit_before_transform():
    norm = Normalizer(mode="zscore")
    with pytest.raises(RuntimeError):
        norm.transform_vector([1.0, 2.0])


def test_normalizer_zscore_amplifies_outliers():
    # benign data tightly clustered; an outlier should get a large z-score
    rows = [[2.0], [2.1], [1.9], [2.0], [2.05]]
    norm = Normalizer(mode="zscore").fit(rows)
    z = norm.transform_vector([10.0])[0]
    assert z > 5.0  # far out of distribution


def test_normalizer_minmax_bounded():
    norm = Normalizer(mode="minmax").fit([[0.0], [10.0]])
    assert norm.transform_vector([5.0])[0] == 0.5
    assert norm.transform_vector([100.0])[0] == 1.0  # clamped
