"""End-to-end pipeline + HTTP API integration."""

import json
import threading
import time
import urllib.request

import pytest

from sentinelx.pipeline import run_pipeline


@pytest.fixture(scope="module")
def service(tmp_path_factory):
    db = str(tmp_path_factory.mktemp("db") / "pipe.db")
    return run_pipeline(db_path=db)


def test_pipeline_summary_quality(service):
    s = service.summary
    assert s["num_windows"] == 40
    assert s["detection"]["precision"] >= 0.9
    assert s["detection"]["recall"] >= 0.6
    assert s["total_incidents"] >= 1


def test_pipeline_is_reproducible(tmp_path):
    a = run_pipeline(db_path=str(tmp_path / "a.db")).summary["detection"]
    b = run_pipeline(db_path=str(tmp_path / "b.db")).summary["detection"]
    assert a == b  # same seed => identical detection metrics


def test_service_network_state(service):
    state = service.network_state()
    assert state["nodes"]
    assert "anomalies" in state
    # last window is an attack window -> at least one anomalous node
    assert any(a["status"] == "anomalous" for a in state["anomalies"])


def test_service_forecast_uncertainty_grows(service):
    fc = service.forecast(window=35, k=3)
    sigmas = [s["uncertainty_sigma"] for s in fc["steps"]]
    assert sigmas == sorted(sigmas)  # non-decreasing with horizon


def test_service_counterfactual_early_window(service):
    # window 30/31: single-source compromise -> isolation should help
    for w in (30, 31):
        state = service.network_state(w)
        anoms = [a["node"] for a in state["anomalies"] if a["status"] == "anomalous"]
        if len(anoms) == 1:
            cf = service.counterfactual("ISOLATE_NODE", window=w, target_node=anoms[0])
            assert cf["delta_risk"] > 0
            return
    pytest.skip("no single-source window found")


def test_service_out_of_range_window(service):
    with pytest.raises(IndexError):
        service.network_state(999)


def test_http_api_endpoints(service):
    port = 8811
    t = threading.Thread(
        target=__import__("sentinelx.api.server", fromlist=["serve"]).serve,
        args=(service,), kwargs={"host": "127.0.0.1", "port": port}, daemon=True,
    )
    t.start()
    time.sleep(0.8)

    def get(path):
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as r:
            return r.status, json.loads(r.read().decode())

    assert get("/health")[1]["status"] == "ok"
    assert get("/summary")[1]["num_windows"] == 40
    assert get("/network/state?window=39")[0] == 200
    assert len(get("/forecast?window=39&k=3")[1]["steps"]) == 3
    assert get("/uncertainty?window=20")[0] == 200
    assert "events" in get("/propagation")[1]
    assert "incidents" in get("/incident")[1]

    # POST counterfactual
    body = json.dumps({"action_type": "ISOLATE_NODE", "window": 30}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/counterfactual", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        cf = json.loads(r.read().decode())
    assert "delta_risk" in cf

    # static dashboard
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as r:
        assert "SENTINEL-X" in r.read().decode()
