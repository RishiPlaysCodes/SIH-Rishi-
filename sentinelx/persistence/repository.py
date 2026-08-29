"""Repository: typed read/write helpers over the SQLite schema.

Everything the pipeline computes is persisted here, and everything the API/
dashboard reads comes back out here, so the two never share in-memory state.
Feature vectors and predicted graphs are stored as JSON text.
"""

from __future__ import annotations

import json
from typing import Any

from sentinelx.analytics.counterfactual import CounterfactualResult
from sentinelx.analytics.novelty import NoveltyResult
from sentinelx.analytics.propagation import PropagationEvent
from sentinelx.analytics.stability import StabilityResult
from sentinelx.analytics.uncertainty import UncertaintyResult
from sentinelx.forecast.deviation import NodeDeviation
from sentinelx.graph.types import GraphState
from sentinelx.persistence.db import Database


class Repository:
    def __init__(self, db: Database):
        self.db = db

    # ---- experiments ------------------------------------------------------ #
    def save_experiment(
        self,
        name: str,
        dataset: str,
        model_type: str,
        config_yaml: str,
        seed: int,
        mlflow_run_id: str | None = None,
    ) -> int:
        return self.db.insert(
            "INSERT INTO experiments(name,dataset,model_type,config_yaml,seed,mlflow_run_id) "
            "VALUES(?,?,?,?,?,?)",
            (name, dataset, model_type, config_yaml, seed, mlflow_run_id),
        )

    def latest_experiment(self) -> dict[str, Any] | None:
        row = self.db.query_one("SELECT * FROM experiments ORDER BY id DESC LIMIT 1")
        return dict(row) if row else None

    # ---- snapshots / nodes / edges --------------------------------------- #
    def save_snapshot(self, experiment_id: int, graph: GraphState) -> int:
        snapshot_id = self.db.insert(
            "INSERT INTO network_snapshots"
            "(experiment_id,window_index,window_start,window_end,node_count,edge_count) "
            "VALUES(?,?,?,?,?,?)",
            (
                experiment_id,
                graph.index,
                graph.window_start,
                graph.window_end,
                graph.node_count(),
                graph.edge_count(),
            ),
        )
        for key in graph.node_keys():
            nd = graph.nodes[key]
            self.db.execute(
                "INSERT INTO nodes(snapshot_id,node_key,label,feature_vector,is_server,status) "
                "VALUES(?,?,?,?,?,?)",
                (snapshot_id, nd.key, nd.label, json.dumps(nd.features), int(nd.is_server), nd.status),
            )
        for e in graph.edges:
            self.db.execute(
                "INSERT INTO edges"
                "(snapshot_id,src_node_key,dst_node_key,protocol,dst_port,feature_vector,weight) "
                "VALUES(?,?,?,?,?,?,?)",
                (snapshot_id, e.src, e.dst, e.protocol, e.dst_port, json.dumps(e.features), e.weight),
            )
        self.db.commit()
        return snapshot_id

    def snapshot_id_for_window(self, experiment_id: int, window_index: int) -> int | None:
        row = self.db.query_one(
            "SELECT id FROM network_snapshots WHERE experiment_id=? AND window_index=?",
            (experiment_id, window_index),
        )
        return int(row["id"]) if row else None

    def list_snapshots(self, experiment_id: int) -> list[dict[str, Any]]:
        rows = self.db.query(
            "SELECT * FROM network_snapshots WHERE experiment_id=? ORDER BY window_index",
            (experiment_id,),
        )
        return [dict(r) for r in rows]

    def get_snapshot_graph(self, snapshot_id: int) -> dict[str, Any]:
        snap = self.db.query_one("SELECT * FROM network_snapshots WHERE id=?", (snapshot_id,))
        nodes = self.db.query("SELECT * FROM nodes WHERE snapshot_id=?", (snapshot_id,))
        edges = self.db.query("SELECT * FROM edges WHERE snapshot_id=?", (snapshot_id,))
        return {
            "snapshot": dict(snap) if snap else None,
            "nodes": [
                {
                    "key": n["node_key"],
                    "label": n["label"],
                    "features": json.loads(n["feature_vector"]),
                    "is_server": bool(n["is_server"]),
                    "status": n["status"],
                }
                for n in nodes
            ],
            "edges": [
                {
                    "src": e["src_node_key"],
                    "dst": e["dst_node_key"],
                    "protocol": e["protocol"],
                    "dst_port": e["dst_port"],
                    "features": json.loads(e["feature_vector"]),
                    "weight": e["weight"],
                }
                for e in edges
            ],
        }

    # ---- anomalies -------------------------------------------------------- #
    def save_anomaly(self, snapshot_id: int, dev: NodeDeviation) -> int:
        return self.db.insert(
            "INSERT INTO anomalies(snapshot_id,node_key,deviation_score,node_state_error,"
            "edge_state_error,feature_pred_error,structural_error,temporal_error,status) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (
                snapshot_id,
                dev.key,
                dev.deviation_score,
                dev.node_state_error,
                dev.edge_state_error,
                dev.feature_pred_error,
                dev.structural_error,
                dev.temporal_error,
                dev.status,
            ),
        )

    def get_anomalies(self, snapshot_id: int) -> list[dict[str, Any]]:
        rows = self.db.query(
            "SELECT * FROM anomalies WHERE snapshot_id=? ORDER BY deviation_score DESC",
            (snapshot_id,),
        )
        return [dict(r) for r in rows]

    # ---- forecasts -------------------------------------------------------- #
    def save_forecast(
        self, experiment_id: int, snapshot_id: int, horizon_step: int, predicted_graph: dict
    ) -> int:
        return self.db.insert(
            "INSERT INTO forecasts(experiment_id,snapshot_id,horizon_step,predicted_graph) "
            "VALUES(?,?,?,?)",
            (experiment_id, snapshot_id, horizon_step, json.dumps(predicted_graph)),
        )

    def save_forecast_result(
        self,
        forecast_id: int,
        uncertainty: UncertaintyResult,
        novelty: NoveltyResult | None = None,
        stability: StabilityResult | None = None,
    ) -> int:
        return self.db.insert(
            "INSERT INTO forecast_results(forecast_id,mean_prediction,std_dev,uncertainty_label,"
            "trajectory_novelty_score,novelty_label,stability_score,stability_label) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (
                forecast_id,
                uncertainty.mean_prediction,
                uncertainty.std_dev,
                uncertainty.label,
                novelty.score if novelty else None,
                novelty.label if novelty else None,
                stability.stability_score if stability else None,
                stability.label if stability else None,
            ),
        )

    def get_forecasts(self, snapshot_id: int) -> list[dict[str, Any]]:
        rows = self.db.query(
            "SELECT f.*, r.mean_prediction, r.std_dev, r.uncertainty_label, "
            "r.trajectory_novelty_score, r.novelty_label, r.stability_score, r.stability_label "
            "FROM forecasts f LEFT JOIN forecast_results r ON r.forecast_id=f.id "
            "WHERE f.snapshot_id=? ORDER BY f.horizon_step",
            (snapshot_id,),
        )
        out = []
        for r in rows:
            d = dict(r)
            d["predicted_graph"] = json.loads(d["predicted_graph"])
            out.append(d)
        return out

    # ---- propagation ------------------------------------------------------ #
    def save_propagation_event(self, snapshot_id: int, ev: PropagationEvent) -> int:
        return self.db.insert(
            "INSERT INTO propagation_events(snapshot_id,source_node_key,target_node_key,"
            "propagation_velocity,propagation_intensity,effective_reproduction_number) "
            "VALUES(?,?,?,?,?,?)",
            (
                snapshot_id,
                ev.source,
                ev.target,
                ev.propagation_velocity,
                ev.propagation_intensity,
                ev.effective_reproduction_number,
            ),
        )

    def get_propagation_events(self, experiment_id: int) -> list[dict[str, Any]]:
        rows = self.db.query(
            "SELECT p.*, s.window_index FROM propagation_events p "
            "JOIN network_snapshots s ON s.id=p.snapshot_id "
            "WHERE s.experiment_id=? ORDER BY s.window_index",
            (experiment_id,),
        )
        return [dict(r) for r in rows]

    # ---- counterfactuals -------------------------------------------------- #
    def save_counterfactual(self, snapshot_id: int, cf: CounterfactualResult) -> int:
        return self.db.insert(
            "INSERT INTO counterfactual_runs(snapshot_id,action_type,target,risk_before,"
            "risk_after,delta_risk) VALUES(?,?,?,?,?,?)",
            (snapshot_id, cf.action_type, cf.target, cf.risk_before, cf.risk_after, cf.delta_risk),
        )

    # ---- incidents / CyberChronicle -------------------------------------- #
    def save_incident(
        self,
        snapshot_id: int,
        event_type: str,
        narrative_text: str,
        mitre_stage: str | None = None,
        contributing_features: list | None = None,
    ) -> int:
        return self.db.insert(
            "INSERT INTO incidents(snapshot_id,event_type,narrative_text,mitre_stage,"
            "contributing_features) VALUES(?,?,?,?,?)",
            (
                snapshot_id,
                event_type,
                narrative_text,
                mitre_stage,
                json.dumps(contributing_features) if contributing_features is not None else None,
            ),
        )

    def get_incidents(self, experiment_id: int) -> list[dict[str, Any]]:
        rows = self.db.query(
            "SELECT i.*, s.window_index, s.window_start FROM incidents i "
            "JOIN network_snapshots s ON s.id=i.snapshot_id "
            "WHERE s.experiment_id=? ORDER BY i.id",
            (experiment_id,),
        )
        out = []
        for r in rows:
            d = dict(r)
            if d.get("contributing_features"):
                d["contributing_features"] = json.loads(d["contributing_features"])
            out.append(d)
        return out
