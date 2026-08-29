"""End-to-end orchestration for Sentinel-X.

Wires every layer together into a single reproducible run:

    config+seed
      -> flows (synthetic or CIC-IDS2018 CSV)
      -> clean -> dynamic graph sequence
      -> leakage-safe temporal split
      -> fit normaliser + world model + novelty scorer on TRAIN ONLY
      -> rolling one-step deviation over the whole sequence
      -> propagation / uncertainty / novelty / stability / k-step forecasts
      -> persist everything to SQLite + emit CyberChronicle incidents

The returned :class:`SentinelService` keeps the fitted model and scored graphs in
memory so the API can answer forecast / counterfactual / uncertainty queries
live, while the SQLite database backs the stored views (network state, anomalies,
incidents, propagation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sentinelx.analytics import (
    NoveltyScorer,
    assess_stability,
    compute_propagation,
    estimate_uncertainty,
)
from sentinelx.analytics.counterfactual import Intervention, run_counterfactual
from sentinelx.analytics.explain import top_feature_names
from sentinelx.analytics.mitre import map_mitre_stage
from sentinelx.config import Config, load_config
from sentinelx.data import (
    Normalizer,
    clean_flows,
    generate_synthetic_flows,
    load_cic_ids_csv,
)
from sentinelx.data.splits import assert_no_leakage, temporal_split
from sentinelx.forecast import ForecastEngine
from sentinelx.forecast.deviation import DeviationResult
from sentinelx.graph import build_graph_sequence
from sentinelx.graph.types import GraphState
from sentinelx.models import build_model
from sentinelx.models.base import WorldModel
from sentinelx.narrative import (
    narrate_deviation,
    narrate_intervention,
    narrate_propagation,
)
from sentinelx.persistence import Database, Repository
from sentinelx.seeding import Rng


@dataclass
class SentinelService:
    """In-memory handle over a completed pipeline run, plus live query methods."""

    config: Config
    experiment_id: int
    graphs: list[GraphState]
    model: WorldModel
    engine: ForecastEngine
    novelty: NoveltyScorer
    normalizer: Normalizer
    devs: list[DeviationResult]
    dev_by_index: dict[int, DeviationResult]
    window_seconds: float
    summary: dict[str, Any]
    repo: Repository
    feature_names: list[str] = field(default_factory=list)

    # ---- live queries used by the API ------------------------------------ #
    def network_state(self, window: int | None = None) -> dict[str, Any]:
        idx = self._resolve_window(window)
        graph = self.graphs[idx]
        dev = self.dev_by_index.get(idx)
        state = graph.to_json()
        anomalies = []
        if dev:
            for key, nd in sorted(
                dev.per_node.items(), key=lambda kv: kv[1].deviation_score, reverse=True
            ):
                anomalies.append(
                    {
                        "node": key,
                        "deviation_score": nd.deviation_score,
                        "status": nd.status,
                        "feature_pred_error": nd.feature_pred_error,
                        "structural_error": nd.structural_error,
                        "temporal_error": nd.temporal_error,
                    }
                )
        state["anomalies"] = anomalies
        state["window"] = idx
        state["num_windows"] = len(self.graphs)
        return state

    def forecast(self, window: int | None = None, k: int | None = None) -> dict[str, Any]:
        idx = self._resolve_window(window)
        k = k or int(self.config.get("forecast.horizon", 3))
        history = self.graphs[: idx + 1]
        preds = self.engine.forecast(history, k)
        rng = Rng(self.config.seed).spawn(f"fc-{idx}")
        steps = []
        for step, g in enumerate(preds, start=1):
            unc = estimate_uncertainty(
                self.model,
                history,
                num_passes=int(self.config.get("uncertainty.num_passes", 30)),
                dropout=float(self.config.get("uncertainty.dropout", 0.2)),
                rng=rng.spawn(f"u{step}"),
                low_sigma=float(self.config.get("uncertainty.low_sigma", 0.30)),
                high_sigma=float(self.config.get("uncertainty.high_sigma", 0.50)),
            )
            # Uncertainty grows with horizon: scale sigma by sqrt(step).
            sigma = unc.std_dev * (step ** 0.5)
            steps.append(
                {
                    "horizon": step,
                    "graph": g.to_json(),
                    "uncertainty_sigma": sigma,
                    "uncertainty_label": unc.label,
                }
            )
            history = list(history) + [g]
        return {"window": idx, "horizon": k, "steps": steps}

    def uncertainty(self, window: int | None = None) -> dict[str, Any]:
        idx = self._resolve_window(window)
        history = self.graphs[: idx + 1]
        rng = Rng(self.config.seed).spawn(f"unc-{idx}")
        result = estimate_uncertainty(
            self.model,
            history,
            num_passes=int(self.config.get("uncertainty.num_passes", 30)),
            dropout=float(self.config.get("uncertainty.dropout", 0.2)),
            rng=rng,
            low_sigma=float(self.config.get("uncertainty.low_sigma", 0.30)),
            high_sigma=float(self.config.get("uncertainty.high_sigma", 0.50)),
        )
        return {
            "window": idx,
            "mean_prediction": result.mean_prediction,
            "std_dev": result.std_dev,
            "label": result.label,
            "per_node_sigma": result.per_node_sigma,
        }

    def propagation(self) -> dict[str, Any]:
        events = compute_propagation(self.graphs, self.dev_by_index, self.window_seconds)
        return {
            "events": [
                {
                    "window": e.window_index,
                    "source": e.source,
                    "target": e.target,
                    "propagation_velocity": e.propagation_velocity,
                    "propagation_intensity": e.propagation_intensity,
                    "effective_reproduction_number": e.effective_reproduction_number,
                }
                for e in events
            ]
        }

    def counterfactual(
        self,
        action_type: str,
        window: int | None = None,
        target_node: str | None = None,
        target_edge: list[str] | None = None,
        port: int | None = None,
        rate_factor: float = 0.2,
    ) -> dict[str, Any]:
        idx = self._resolve_window(window)
        history = self.graphs[: idx + 1]
        dev = self.dev_by_index.get(idx)
        compromised = dev.anomalous_keys() + dev.deviating_keys() if dev else []
        if not compromised and target_node:
            compromised = [target_node]
        iv = Intervention(
            action_type=action_type,
            target_node=target_node,
            target_edge=tuple(target_edge) if target_edge else None,
            port=port,
            rate_factor=rate_factor,
        )
        cf = run_counterfactual(
            self.model, history, iv, compromised,
            horizon=int(self.config.get("forecast.horizon", 3)),
        )
        snapshot_id = self.repo.snapshot_id_for_window(self.experiment_id, idx)
        if snapshot_id:
            self.repo.save_counterfactual(snapshot_id, cf)
            self.repo.save_incident(
                snapshot_id,
                "intervention_simulated",
                narrate_intervention(self.graphs[idx].window_start, cf),
            )
        return {
            "window": idx,
            "action_type": cf.action_type,
            "target": cf.target,
            "risk_before": cf.risk_before,
            "risk_after": cf.risk_after,
            "delta_risk": cf.delta_risk,
            "components_before": cf.components_before,
            "components_after": cf.components_after,
        }

    def incidents(self) -> dict[str, Any]:
        return {"incidents": self.repo.get_incidents(self.experiment_id)}

    def _resolve_window(self, window: int | None) -> int:
        if window is None:
            return len(self.graphs) - 1
        if not (0 <= window < len(self.graphs)):
            raise IndexError(f"window {window} out of range [0,{len(self.graphs)})")
        return window


def _load_flows(cfg: Config):
    data = cfg.section("data")
    dataset = cfg.get("experiment.dataset", "synthetic")
    if dataset == "synthetic":
        return generate_synthetic_flows(
            num_windows=int(data.get("num_windows", 40)),
            window_seconds=int(data.get("window_seconds", 60)),
            num_hosts=int(data.get("num_hosts", 18)),
            num_servers=int(data.get("num_servers", 4)),
            attack_start_window=int(data.get("attack_start_window", 30)),
            attack_type=str(data.get("attack_type", "lateral_movement")),
            seed=cfg.seed,
        )
    # Otherwise treat `dataset` as a path to a CIC-IDS2018-style CSV.
    return load_cic_ids_csv(dataset, is_path=True)


def _attack_nodes_by_window(flows, window_seconds: float) -> dict[int, set]:
    """Ground-truth attacker sources per window (labels used ONLY for eval)."""
    if not flows:
        return {}
    # Match the grid anchoring used by build_windows so window indices align.
    t_min = (min(f.ts for f in flows) // window_seconds) * window_seconds
    by_window: dict[int, set] = {}
    for f in flows:
        if f.is_attack:
            idx = int((f.ts - t_min) // window_seconds)
            by_window.setdefault(idx, set()).add(f.src)
    return by_window


def run_pipeline(
    config_path: str | None = None,
    overrides: dict[str, Any] | None = None,
    db_path: str | None = None,
    reset_db: bool = True,
) -> SentinelService:
    cfg = load_config(config_path, overrides)
    window_seconds = float(cfg.get("data.window_seconds", 60))

    flows = clean_flows(_load_flows(cfg))
    graphs = build_graph_sequence(
        flows, int(window_seconds), float(cfg.get("graph.min_edge_weight", 1.0))
    )
    if len(graphs) < 3:
        raise ValueError("Need at least 3 graph windows to run the pipeline")

    split = temporal_split(len(graphs), float(cfg.get("data.test_fraction", 0.3)))
    assert_no_leakage(split)

    # Fit the normaliser on TRAIN windows only, then apply to every graph.
    train_rows: list[list[float]] = []
    for i in split.train_indices:
        train_rows.extend(graphs[i].feature_matrix())
    normalizer = Normalizer(mode="zscore").fit(train_rows)
    for g in graphs:
        for node in g.nodes.values():
            node.features = normalizer.transform_vector(node.features)

    train_graphs = [graphs[i] for i in split.train_indices]
    model_type = cfg.get("experiment.model_type", "linear_transition")
    model = build_model(model_type, cfg.section("model")).fit(train_graphs)
    novelty = NoveltyScorer(**cfg.section("novelty")).fit(train_graphs)

    dev_cfg = cfg.section("deviation")
    engine = ForecastEngine(
        model,
        weights=dev_cfg.get("weights"),
        anomaly_threshold=float(dev_cfg.get("anomaly_threshold", 0.45)),
        deviating_threshold=float(dev_cfg.get("deviating_threshold", 0.25)),
    )
    devs = engine.rolling_deviation(graphs)
    dev_by_index = {d.graph_index: d for d in devs}
    for d in devs:
        engine.apply_statuses(graphs[d.graph_index], d)

    # ---- persistence ----------------------------------------------------- #
    db = Database(db_path or cfg.get("persistence.db_path", "sentinelx.db"))
    if reset_db:
        db.reset()
    repo = Repository(db)
    experiment_id = repo.save_experiment(
        name=cfg.get("experiment.name", "sentinelx"),
        dataset=cfg.get("experiment.dataset", "synthetic"),
        model_type=model_type,
        config_yaml=cfg.to_yaml(),
        seed=cfg.seed,
    )
    service = SentinelService(
        config=cfg,
        experiment_id=experiment_id,
        graphs=graphs,
        model=model,
        engine=engine,
        novelty=novelty,
        normalizer=normalizer,
        devs=devs,
        dev_by_index=dev_by_index,
        window_seconds=window_seconds,
        summary={},
        repo=repo,
        feature_names=graphs[0].node_feature_names,
    )

    _persist_run(service, split)
    service.summary = _evaluate(service, flows, split)
    return service


def _persist_run(service: SentinelService, split) -> None:
    repo = service.repo
    graphs = service.graphs
    feature_names = service.feature_names
    horizon = int(service.config.get("forecast.horizon", 3))

    snapshot_ids: dict[int, int] = {}
    for g in graphs:
        snapshot_ids[g.index] = repo.save_snapshot(service.experiment_id, g)

    # Anomalies + CyberChronicle deviation/forecast incidents.
    for d in service.devs:
        sid = snapshot_ids[d.graph_index]
        for nd in d.per_node.values():
            repo.save_anomaly(sid, nd)
        top = max(d.per_node.values(), key=lambda x: x.deviation_score, default=None)
        if top and top.status == "anomalous":
            history = graphs[: d.graph_index]
            if history:
                pred = service.model.predict_next(history)
                actual = graphs[d.graph_index]
                if top.key in pred.nodes and top.key in actual.nodes:
                    tops = top_feature_names(
                        pred.nodes[top.key].features,
                        actual.nodes[top.key].features,
                        feature_names,
                        3,
                    )
                else:
                    tops = []
                stage = map_mitre_stage(tops)
                repo.save_incident(
                    sid,
                    "deviation_detected",
                    narrate_deviation(
                        graphs[d.graph_index].window_start, top.key, top.deviation_score, tops, stage
                    ),
                    mitre_stage=stage,
                    contributing_features=tops,
                )

    # Propagation events + incidents.
    events = compute_propagation(graphs, service.dev_by_index, service.window_seconds)
    for e in events:
        sid = snapshot_ids.get(e.window_index)
        if sid:
            repo.save_propagation_event(sid, e)
            repo.save_incident(
                sid, "propagation_detected", narrate_propagation(graphs[e.window_index].window_start, e)
            )

    # K-step forecasts + uncertainty results for each snapshot.
    rng = Rng(service.config.seed).spawn("persist-fc")
    for g in graphs:
        history = graphs[: g.index + 1]
        preds = service.engine.forecast(history, horizon)
        unc = estimate_uncertainty(
            service.model, history,
            num_passes=int(service.config.get("uncertainty.num_passes", 30)),
            dropout=float(service.config.get("uncertainty.dropout", 0.2)),
            rng=rng.spawn(f"w{g.index}"),
            low_sigma=float(service.config.get("uncertainty.low_sigma", 0.30)),
            high_sigma=float(service.config.get("uncertainty.high_sigma", 0.50)),
        )
        nov = service.novelty.score(
            g, service.dev_by_index[g.index].graph_score if g.index in service.dev_by_index else 0.0,
            unc.std_dev,
        )
        stab = assess_stability(
            service.model, history,
            perturbation=float(service.config.get("stability.perturbation", 0.03)),
            num_trials=int(service.config.get("stability.num_trials", 12)),
            unstable_threshold=float(service.config.get("stability.unstable_threshold", 0.12)),
            rng=rng.spawn(f"stab{g.index}"),
        )
        sid = snapshot_ids[g.index]
        for step, pg in enumerate(preds, start=1):
            fid = repo.save_forecast(service.experiment_id, sid, step, pg.to_json())
            if step == 1:
                repo.save_forecast_result(fid, unc, nov, stab)


def _evaluate(service: SentinelService, flows, split) -> dict[str, Any]:
    """Precision/recall of anomalous detection vs flow labels (eval only)."""
    truth = _attack_nodes_by_window(flows, service.window_seconds)
    tp = fp = fn = 0
    for d in service.devs:
        attack_set = truth.get(d.graph_index, set())
        flagged = set(d.anomalous_keys())
        tp += len(flagged & attack_set)
        fp += len(flagged - attack_set)
        fn += len(attack_set - flagged)
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    return {
        "experiment_id": service.experiment_id,
        "num_windows": len(service.graphs),
        "train_windows": len(split.train_indices),
        "test_windows": len(split.test_indices),
        "boundary": split.boundary,
        "model_type": service.config.get("experiment.model_type"),
        "detection": {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
        },
        "total_incidents": len(service.repo.get_incidents(service.experiment_id)),
    }
