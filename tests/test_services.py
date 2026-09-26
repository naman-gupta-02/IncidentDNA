"""The Dockerised microservices, exercised in-process.

These run the real service code — real HTTP handlers, real instrumentation,
real fault controller — without Docker, Kafka or Postgres, so the container
path is covered by CI rather than only by `docker compose up`.
"""
import importlib.util
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from incidentdna import topology as topo
from incidentdna.telemetry.schema import SPAN_KIND_CLIENT, SPAN_KIND_SERVER, Trace


def load_service(directory: str):
    """Import a service's `main.py`. The directories are hyphenated (as the
    design document specifies) so they are not importable as packages."""
    path = ROOT / "services" / directory / "main.py"
    spec = importlib.util.spec_from_file_location(f"svc_{directory.replace('-', '_')}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def inventory():
    module = load_service("inventory-service")
    with TestClient(module.app) as client:
        yield module, client


@pytest.fixture(scope="module")
def payment():
    module = load_service("payment-service")
    with TestClient(module.app) as client:
        yield module, client


def test_inventory_serves_products(inventory):
    _, client = inventory
    body = client.get("/products/3").json()
    assert body["id"] == 3 and body["price_cents"] > 0


def test_inventory_check_reports_availability(inventory):
    _, client = inventory
    body = client.get("/check/3?quantity=2").json()
    assert body["available"] is True and body["price_cents"] > 0


def test_payment_authorizes(payment):
    _, client = payment
    body = client.post("/authorize", json={"user_id": 1, "amount_cents": 500}).json()
    assert "authorized" in body


def test_health_and_fault_endpoints_exist(inventory):
    _, client = inventory
    assert client.get("/health").json()["status"] == "ok"
    assert "database_latency" in client.get("/fault").json()["available"]


def test_fault_injection_round_trip(inventory):
    module, client = inventory
    r = client.post("/fault/cpu_saturation", json={"severity": 0.5, "duration_seconds": 20})
    assert r.status_code == 200 and r.json()["name"] == "cpu_saturation"
    assert client.get("/fault").json()["active"]
    assert client.request("DELETE", "/fault").json()["cleared"] >= 1
    assert not client.get("/fault").json()["active"]


def test_unknown_fault_is_rejected(inventory):
    _, client = inventory
    assert client.post("/fault/not_real", json={}).status_code == 400


def test_bad_deployment_makes_requests_fail(inventory):
    module, client = inventory
    client.post("/fault/bad_deployment", json={"severity": 1.0, "duration_seconds": 60})
    time.sleep(0.2)   # let the ramp leave zero
    statuses = {client.get("/products/3").status_code for _ in range(60)}
    client.request("DELETE", "/fault")
    assert 500 in statuses, "a bad deployment must actually break some requests"


def test_payment_timeout_fault_actually_delays(payment):
    module, client = payment
    baseline = time.perf_counter()
    client.post("/authorize", json={"user_id": 1, "amount_cents": 100})
    normal = time.perf_counter() - baseline

    client.post("/fault/payment_timeout", json={"severity": 1.0, "duration_seconds": 30})
    time.sleep(6)     # ramp to full intensity
    started = time.perf_counter()
    client.post("/authorize", json={"user_id": 1, "amount_cents": 100})
    slow = time.perf_counter() - started
    client.request("DELETE", "/fault")
    assert slow > normal + 0.4


def test_the_request_path_emits_a_well_formed_trace(inventory):
    module, client = inventory
    module.ctx.telemetry.flush()
    with module.ctx.telemetry._lock:
        module.ctx.telemetry._pending.spans.clear()

    import services.common.telemetry as T

    original, T.TRACE_SAMPLE_RATE = T.TRACE_SAMPLE_RATE, 1.0
    try:
        client.get("/products/4")
    finally:
        T.TRACE_SAMPLE_RATE = original

    with module.ctx.telemetry._lock:
        spans = list(module.ctx.telemetry._pending.spans)
    assert spans, "a sampled request must emit a span"
    span = spans[-1]
    assert span.service == topo.INVENTORY_SERVICE
    assert span.kind == SPAN_KIND_SERVER
    assert span.duration_ms > 0
    assert len(span.trace_id) == 32


def test_traceparent_is_parsed_and_propagated():
    from services.common.telemetry import format_traceparent, parse_traceparent

    header = format_traceparent("a" * 32, "b" * 16, True)
    trace_id, parent, sampled = parse_traceparent(header)
    assert trace_id == "a" * 32 and parent == "b" * 16 and sampled is True
    assert parse_traceparent(None) == (None, None, None)
    assert parse_traceparent("garbage") == (None, None, None)


def test_incoming_trace_context_is_adopted(inventory):
    module, client = inventory
    from services.common.telemetry import format_traceparent

    trace_id = "c" * 32
    response = client.get(
        "/products/5", headers={"traceparent": format_traceparent(trace_id, "d" * 16, True)}
    )
    assert response.headers["x-trace-id"] == trace_id


def test_counters_are_flushed_as_metric_points(inventory):
    module, client = inventory
    for _ in range(3):
        client.get("/products/2")
    module.ctx.telemetry.flush()
    # The no-op bus discards them, so assert on what was staged before flush.
    for _ in range(3):
        client.get("/products/2")
    with module.ctx.telemetry._lock:
        names = {m.metric for m in module.ctx.telemetry._pending.metrics}
    assert "http_request_duration_ms" in names
    assert "self_duration_ms" in names
