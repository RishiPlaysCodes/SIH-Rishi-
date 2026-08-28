"""SQLite persistence + repository round-trips."""

from sentinelx.forecast.deviation import NodeDeviation
from sentinelx.persistence import Database, Repository


def test_schema_creates_all_tables(tmp_db):
    db = Database(tmp_db)
    rows = db.query("SELECT name FROM sqlite_master WHERE type='table'")
    names = {r["name"] for r in rows}
    expected = {
        "experiments", "network_snapshots", "nodes", "edges", "anomalies",
        "forecasts", "forecast_results", "propagation_events",
        "counterfactual_runs", "incidents",
    }
    assert expected.issubset(names)
    db.close()


def test_experiment_and_snapshot_roundtrip(tmp_db, graph_sequence):
    db = Database(tmp_db)
    repo = Repository(db)
    exp_id = repo.save_experiment("t", "synthetic", "linear_transition", "cfg: 1", 1337)
    assert exp_id > 0
    assert repo.latest_experiment()["id"] == exp_id

    sid = repo.save_snapshot(exp_id, graph_sequence[0])
    graph = repo.get_snapshot_graph(sid)
    assert graph["snapshot"]["node_count"] == graph_sequence[0].node_count()
    assert len(graph["nodes"]) == graph_sequence[0].node_count()
    assert len(graph["edges"]) == graph_sequence[0].edge_count()
    db.close()


def test_anomaly_and_incident_persist(tmp_db, graph_sequence):
    db = Database(tmp_db)
    repo = Repository(db)
    exp_id = repo.save_experiment("t", "synthetic", "linear_transition", "cfg", 1)
    sid = repo.save_snapshot(exp_id, graph_sequence[0])
    dev = NodeDeviation("HOST-01", 0.8, 0.5, 0.1, 0.2, 0.3, 0.4, "anomalous")
    repo.save_anomaly(sid, dev)
    anomalies = repo.get_anomalies(sid)
    assert anomalies[0]["node_key"] == "HOST-01"
    assert anomalies[0]["deviation_score"] == 0.8

    repo.save_incident(sid, "deviation_detected", "something happened",
                       mitre_stage="Lateral Movement", contributing_features=["a", "b"])
    incidents = repo.get_incidents(exp_id)
    assert incidents[0]["mitre_stage"] == "Lateral Movement"
    assert incidents[0]["contributing_features"] == ["a", "b"]
    db.close()


def test_reset_clears_data(tmp_db, graph_sequence):
    db = Database(tmp_db)
    repo = Repository(db)
    repo.save_experiment("t", "synthetic", "m", "cfg", 1)
    db.reset()
    assert repo.latest_experiment() is None
    db.close()
