-- Sentinel-X SQLite schema (PRD §4).
-- Foreign keys are declared and enforced (PRAGMA foreign_keys = ON in db.py).

CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    dataset TEXT NOT NULL,
    model_type TEXT NOT NULL,
    config_yaml TEXT NOT NULL,
    seed INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    mlflow_run_id TEXT
);

CREATE TABLE IF NOT EXISTS network_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER REFERENCES experiments(id),
    window_index INTEGER NOT NULL,
    window_start TIMESTAMP NOT NULL,
    window_end TIMESTAMP NOT NULL,
    node_count INTEGER,
    edge_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS nodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id INTEGER REFERENCES network_snapshots(id),
    node_key TEXT NOT NULL,
    label TEXT,
    feature_vector TEXT NOT NULL,
    is_server INTEGER DEFAULT 0,
    status TEXT DEFAULT 'normal'
);

CREATE TABLE IF NOT EXISTS edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id INTEGER REFERENCES network_snapshots(id),
    src_node_key TEXT NOT NULL,
    dst_node_key TEXT NOT NULL,
    protocol TEXT,
    dst_port INTEGER,
    feature_vector TEXT NOT NULL,
    weight REAL
);

CREATE TABLE IF NOT EXISTS anomalies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id INTEGER REFERENCES network_snapshots(id),
    node_key TEXT NOT NULL,
    deviation_score REAL NOT NULL,
    node_state_error REAL,
    edge_state_error REAL,
    feature_pred_error REAL,
    structural_error REAL,
    temporal_error REAL,
    status TEXT,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS forecasts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER REFERENCES experiments(id),
    snapshot_id INTEGER REFERENCES network_snapshots(id),
    horizon_step INTEGER NOT NULL,
    predicted_graph TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS forecast_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    forecast_id INTEGER REFERENCES forecasts(id),
    mean_prediction REAL,
    std_dev REAL,
    uncertainty_label TEXT,
    trajectory_novelty_score REAL,
    novelty_label TEXT,
    stability_score REAL,
    stability_label TEXT
);

CREATE TABLE IF NOT EXISTS propagation_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id INTEGER REFERENCES network_snapshots(id),
    source_node_key TEXT,
    target_node_key TEXT,
    propagation_velocity REAL,
    propagation_intensity REAL,
    effective_reproduction_number REAL,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS counterfactual_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id INTEGER REFERENCES network_snapshots(id),
    action_type TEXT NOT NULL,
    target TEXT,
    risk_before REAL,
    risk_after REAL,
    delta_risk REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS incidents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id INTEGER REFERENCES network_snapshots(id),
    event_type TEXT NOT NULL,
    narrative_text TEXT NOT NULL,
    mitre_stage TEXT,
    contributing_features TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_snapshots_experiment ON network_snapshots(experiment_id);
CREATE INDEX IF NOT EXISTS idx_nodes_snapshot ON nodes(snapshot_id);
CREATE INDEX IF NOT EXISTS idx_edges_snapshot ON edges(snapshot_id);
CREATE INDEX IF NOT EXISTS idx_anomalies_snapshot ON anomalies(snapshot_id);
CREATE INDEX IF NOT EXISTS idx_incidents_snapshot ON incidents(snapshot_id);
