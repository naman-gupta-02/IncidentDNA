"""The HTTP surface the dashboard depends on."""
import os
import time

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("INCIDENTDNA_SPEED", "60")
os.environ.setdefault("INCIDENTDNA_SEED", "7")

from backend.app import app
from incidentdna import topology as topo


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["windows_processed"] > 0, "the system warms up before serving"


def test_topology_matches_the_graph(client):
    body = client.get("/api/topology").json()
    assert {s["name"] for s in body["services"]} == set(topo.SERVICES)
    assert len(body["edges"]) == len(topo.EDGES)


def test_state_reports_every_service(client):
    body = client.get("/api/state").json()
    assert {s["service"] for s in body["services"]} == set(topo.SERVICES)
    assert body["ranker"] in body["available_rankers"]
    for s in body["services"]:
        assert s["health"] in {"healthy", "degraded", "critical"}


def test_fault_catalogue_is_served(client):
    body = client.get("/api/faults").json()
    names = {f["name"] for f in body}
    assert "database_latency" in names
    for f in body:
        assert f["origins"] and f["mechanism"]


def test_metrics_endpoint_returns_aligned_series(client):
    body = client.get("/api/metrics?services=postgres,order-service&limit=20").json()
    assert set(body["metrics"]) == {"postgres", "order-service"}
    pg = body["metrics"]["postgres"]
    assert len(pg["window_index"]) == len(pg["latency_p95"])


def test_metrics_rejects_an_unknown_service(client):
    assert client.get("/api/metrics?services=not-a-service").status_code == 400


def test_unknown_incident_is_404(client):
    assert client.get("/api/incidents/inc_99999").status_code == 404


def test_unknown_ranker_is_rejected(client):
    assert client.post("/api/ranker", json={"ranker": "nope"}).status_code == 400


def test_unknown_fault_is_rejected(client):
    r = client.post("/api/faults/inject", json={"fault_type": "not_a_fault"})
    assert r.status_code == 400


def test_fault_cannot_originate_in_an_impossible_service(client):
    r = client.post(
        "/api/faults/inject",
        json={"fault_type": "database_latency", "origin": topo.PAYMENT_SERVICE},
    )
    assert r.status_code == 400


def test_injecting_a_fault_produces_a_diagnosed_incident(client):
    r = client.post(
        "/api/faults/inject",
        json={"fault_type": "database_latency", "severity": 0.9, "duration_seconds": 300},
    )
    assert r.status_code == 200 and r.json()["origin"] == topo.POSTGRES

    incident_id = None
    for _ in range(60):
        time.sleep(0.3)
        incident_id = client.get("/api/state").json()["open_incident_id"]
        if incident_id:
            break
    assert incident_id, "no incident opened after injecting a database fault"

    # Give the ranker a couple of windows of evidence.
    for _ in range(20):
        time.sleep(0.3)
        incident = client.get(f"/api/incidents/{incident_id}").json()
        if incident.get("diagnosis"):
            break
    assert incident["diagnosis"]["root_cause"] == topo.POSTGRES
    assert incident["diagnosis"]["candidates"][0]["evidence"]
    assert incident["symptom"]["text"]

    summary = client.get(f"/api/incidents/{incident_id}/summary").text
    assert "Most likely root cause" in summary

    listed = client.get("/api/incidents").json()
    assert any(i["incident_id"] == incident_id for i in listed)

    client.post("/api/faults/clear")


def test_dashboard_is_served(client):
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200
