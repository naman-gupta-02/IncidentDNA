"""Trace maths: self-time attribution is the basis of trace evidence."""
from incidentdna.telemetry.schema import SPAN_KIND_CLIENT, SPAN_KIND_SERVER, Span, Trace


def _span(sid, parent, service, kind, dur, start=0.0, peer=None, status="OK"):
    return Span(
        trace_id="t1", span_id=sid, parent_span_id=parent, service=service,
        operation="op", kind=kind, start_time=start, duration_ms=dur,
        peer_service=peer, status=status,
    )


def _db_trace():
    """gateway 2400ms -> order 2350ms -> postgres 2200ms."""
    return Trace("t1", [
        _span("a", None, "api-gateway", SPAN_KIND_SERVER, 2400),
        _span("b", "a", "api-gateway", SPAN_KIND_CLIENT, 2380, peer="order-service"),
        _span("c", "b", "order-service", SPAN_KIND_SERVER, 2350),
        _span("d", "c", "order-service", SPAN_KIND_CLIENT, 2200, peer="postgres"),
        _span("e", "d", "postgres", SPAN_KIND_SERVER, 2200),
    ])


def test_self_time_credits_the_database_not_the_gateway():
    st = _db_trace().self_time_ms()
    assert st["postgres"] == 2200
    assert st["api-gateway"] == 2400 - 2380
    assert st["order-service"] == 2350 - 2200


def test_dominant_service_is_the_cause_not_the_symptom():
    assert _db_trace().dominant_service() == "postgres"


def test_failed_is_true_when_any_span_failed():
    trace = _db_trace()
    assert not trace.failed
    trace.spans[-1].status = "TIMEOUT"
    assert trace.failed


def test_trace_round_trips_to_a_dict():
    payload = _db_trace().to_dict()
    assert payload["dominant_service"] == "postgres"
    assert len(payload["spans"]) == 5
