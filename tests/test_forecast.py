"""Forecast engine + behavioural deviation scoring + detection quality."""

from sentinelx.data import Normalizer, temporal_split
from sentinelx.forecast import ForecastEngine, compute_deviation
from sentinelx.forecast.deviation import saturate
from sentinelx.models import build_model


def _prepare(graph_sequence, model_type="linear_transition"):
    graphs = [g.clone() for g in graph_sequence]
    split = temporal_split(len(graphs), 0.3)
    rows = []
    for i in split.train_indices:
        rows.extend(graphs[i].feature_matrix())
    norm = Normalizer(mode="zscore").fit(rows)
    for g in graphs:
        for nd in g.nodes.values():
            nd.features = norm.transform_vector(nd.features)
    model = build_model(model_type, {"ridge_lambda": 0.05}).fit(
        [graphs[i] for i in split.train_indices]
    )
    return graphs, model, split


def test_saturate_bounds():
    assert saturate(0.0, 2.5) == 0.0
    assert 0.0 < saturate(2.5, 2.5) < 1.0
    assert saturate(1e9, 2.5) < 1.0


def test_deviation_zero_when_identical(graph_sequence):
    g = graph_sequence[5]
    result = compute_deviation(g, g, previous=g)
    assert result.graph_score < 1e-6


def test_rolling_deviation_covers_all_but_first(graph_sequence):
    graphs, model, _ = _prepare(graph_sequence)
    engine = ForecastEngine(model)
    devs = engine.rolling_deviation(graphs)
    assert len(devs) == len(graphs) - 1
    assert devs[0].graph_index == 1


def test_attack_scores_higher_than_benign(graph_sequence):
    graphs, model, _ = _prepare(graph_sequence)
    engine = ForecastEngine(model)
    devs = engine.rolling_deviation(graphs)
    benign_max = max(
        v.deviation_score for d in devs if d.graph_index < 30 for v in d.per_node.values()
    )
    attack_top = min(
        max(d.per_node.values(), key=lambda x: x.deviation_score).deviation_score
        for d in devs
        if d.graph_index >= 32  # after ramp-up
    )
    assert attack_top > benign_max * 1.5


def test_detection_precision_recall(graph_sequence):
    graphs, model, _ = _prepare(graph_sequence)
    engine = ForecastEngine(model)
    devs = engine.rolling_deviation(graphs)
    # ground truth: attacker sources per window
    from sentinelx.data import clean_flows, generate_synthetic_flows
    from sentinelx.pipeline import _attack_nodes_by_window

    flows = clean_flows(
        generate_synthetic_flows(num_windows=40, num_hosts=14, num_servers=4,
                                 attack_start_window=30, seed=1337)
    )
    truth = _attack_nodes_by_window(flows, 60)
    tp = fp = fn = 0
    for d in devs:
        atk = truth.get(d.graph_index, set())
        flagged = set(d.anomalous_keys())
        tp += len(flagged & atk)
        fp += len(flagged - atk)
        fn += len(atk - flagged)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    assert precision >= 0.9
    assert recall >= 0.6


def test_forecast_horizon_validation(graph_sequence):
    graphs, model, _ = _prepare(graph_sequence)
    engine = ForecastEngine(model)
    import pytest

    with pytest.raises(ValueError):
        engine.forecast(graphs[:10], 0)
