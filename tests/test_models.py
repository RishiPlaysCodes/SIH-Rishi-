"""World models and the forecasting interface."""

import pytest

from sentinelx.models import (
    EWMAModel,
    LinearTransitionModel,
    PersistenceModel,
    available_models,
    build_model,
)
from sentinelx.models.base import apply_dropout
from sentinelx.seeding import Rng


def test_registry_builds_all(graph_sequence):
    for name in available_models():
        model = build_model(name, {"ridge_lambda": 0.05, "ewma_alpha": 0.4})
        model.fit(graph_sequence[:20])
        pred = model.predict_next(graph_sequence[:20])
        assert pred.node_count() > 0


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        build_model("transformer_xl")


def test_persistence_predicts_last_state(graph_sequence):
    model = PersistenceModel()
    last = graph_sequence[10]
    pred = model.predict_next(graph_sequence[:11])
    for key in last.nodes:
        assert pred.nodes[key].features == last.nodes[key].features


def test_predict_next_increments_index(graph_sequence):
    model = PersistenceModel()
    pred = model.predict_next(graph_sequence[:5])
    assert pred.index == graph_sequence[4].index + 1


def test_predict_sequence_length(graph_sequence):
    model = LinearTransitionModel().fit(graph_sequence[:20])
    preds = model.predict_sequence(graph_sequence[:20], k=3)
    assert len(preds) == 3
    assert [p.index for p in preds] == [20, 21, 22]


def test_linear_model_learns_nontrivial_map(graph_sequence):
    model = LinearTransitionModel(ridge_lambda=0.01).fit(graph_sequence[:25])
    assert model._fitted
    assert len(model.W) == 9  # 8 features + bias


def test_ewma_requires_valid_alpha():
    with pytest.raises(ValueError):
        EWMAModel(alpha=0.0)


def test_apply_dropout_is_deterministic_with_seed():
    rng1, rng2 = Rng(5), Rng(5)
    v = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert apply_dropout(v, 0.5, rng1) == apply_dropout(v, 0.5, rng2)


def test_apply_dropout_zero_is_identity():
    v = [1.0, 2.0, 3.0]
    assert apply_dropout(v, 0.0, None) == v


def test_predict_next_empty_history_raises():
    with pytest.raises(ValueError):
        PersistenceModel().predict_next([])
