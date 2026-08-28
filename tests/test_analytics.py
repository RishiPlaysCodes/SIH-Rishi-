"""Analytics engines: propagation, uncertainty, novelty, stability, CF, MITRE."""

import pytest

from sentinelx.analytics import (
    NoveltyScorer,
    assess_stability,
    compute_propagation,
    estimate_uncertainty,
    feature_contributions,
    map_mitre_stage,
)
from sentinelx.analytics.counterfactual import Intervention, apply_intervention, run_counterfactual
from sentinelx.data import Normalizer, temporal_split
from sentinelx.forecast import ForecastEngine
from sentinelx.models import build_model
from sentinelx.seeding import Rng


def _scored(graph_sequence):
    graphs = [g.clone() for g in graph_sequence]
    split = temporal_split(len(graphs), 0.3)
    rows = []
    for i in split.train_indices:
        rows.extend(graphs[i].feature_matrix())
    norm = Normalizer(mode="zscore").fit(rows)
    for g in graphs:
        for nd in g.nodes.values():
            nd.features = norm.transform_vector(nd.features)
    model = build_model("linear_transition", {"ridge_lambda": 0.05}).fit(
        [graphs[i] for i in split.train_indices]
    )
    engine = ForecastEngine(model)
    devs = engine.rolling_deviation(graphs)
    dev_by = {d.graph_index: d for d in devs}
    for d in devs:
        engine.apply_statuses(graphs[d.graph_index], d)
    return graphs, model, engine, dev_by, split


def test_propagation_detects_chain(graph_sequence):
    graphs, _, _, dev_by, _ = _scored(graph_sequence)
    events = compute_propagation(graphs, dev_by, 60)
    assert len(events) >= 1
    for e in events:
        assert e.effective_reproduction_number >= 0
        assert e.source != e.target


def test_uncertainty_grows_or_labels(graph_sequence):
    graphs, model, _, _, _ = _scored(graph_sequence)
    benign = estimate_uncertainty(model, graphs[:20], num_passes=25, rng=Rng(1))
    attack = estimate_uncertainty(model, graphs[:39], num_passes=25, rng=Rng(1))
    assert benign.std_dev >= 0
    assert attack.std_dev > benign.std_dev  # attack context is less certain


def test_uncertainty_requires_multiple_passes(graph_sequence):
    graphs, model, _, _, _ = _scored(graph_sequence)
    with pytest.raises(ValueError):
        estimate_uncertainty(model, graphs[:5], num_passes=1)


def test_novelty_attack_more_novel_than_benign(graph_sequence):
    graphs, model, engine, dev_by, split = _scored(graph_sequence)
    scorer = NoveltyScorer().fit([graphs[i] for i in split.train_indices])
    benign = scorer.score(graphs[10], dev_by[10].graph_score, 0.1)
    attack = scorer.score(graphs[38], dev_by[38].graph_score, 0.5)
    assert attack.score > benign.score


def test_stability_is_bounded(graph_sequence):
    graphs, model, _, _, _ = _scored(graph_sequence)
    result = assess_stability(model, graphs[:20], num_trials=8, rng=Rng(2))
    assert 0.0 < result.stability_score <= 1.0
    assert result.label in ("STABLE", "UNSTABLE")


def test_counterfactual_isolate_reduces_risk_early(graph_sequence):
    graphs, model, engine, dev_by, _ = _scored(graph_sequence)
    # find first window with exactly one anomalous node
    target_win = None
    for idx in range(30, 40):
        if idx in dev_by:
            anoms = dev_by[idx].anomalous_keys()
            if len(anoms) == 1:
                target_win = idx
                target = anoms[0]
                break
    assert target_win is not None
    cf = run_counterfactual(
        model, graphs[: target_win + 1],
        Intervention("ISOLATE_NODE", target_node=target),
        compromised=[target], horizon=3,
    )
    assert cf.delta_risk > 0  # isolating the sole source reduces risk


def test_apply_intervention_isolate_removes_node(graph_sequence):
    g = graph_sequence[35]
    key = g.node_keys()[0]
    g2 = apply_intervention(g, Intervention("ISOLATE_NODE", target_node=key))
    assert key not in g2.nodes
    assert all(e.src != key and e.dst != key for e in g2.edges)


def test_invalid_action_raises():
    with pytest.raises(ValueError):
        Intervention("NUKE_EVERYTHING")


def test_mitre_mapping():
    assert map_mitre_stage(["unique_destinations"]) == "Lateral Movement"
    assert map_mitre_stage(["mean_byte_rate"]) == "Exfiltration"
    assert map_mitre_stage([]) == "Unknown"
    # prefers more advanced stage among top-2
    stage = map_mitre_stage(["failed_connections", "mean_byte_rate"])
    assert stage == "Exfiltration"


def test_feature_contributions_sum_to_one():
    contribs = feature_contributions([1.0, 0.0, 0.0], [0.0, 0.0, 0.0], ["a", "b", "c"])
    total = sum(c["contribution"] for c in contribs)
    assert abs(total - 1.0) < 1e-9
    assert contribs[0]["feature"] == "a"  # largest residual first
