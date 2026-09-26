"""Stage two: the features that separate a cause from a symptom."""
import pytest

from incidentdna import topology as topo
from incidentdna.config import WINDOW_SECONDS
from incidentdna.incidents.models import Incident, ServiceIncidentState
from incidentdna.pipeline.features import ServiceWindow
from incidentdna.pipeline.store import FeatureStore
from incidentdna.rca.baseline import (
    AnomalyStrengthScorer,
    CausalLocalityScorer,
    TransparentScorer,
    softmax_confidence,
)
from incidentdna.rca.candidates import IncidentContext, build_candidates
from incidentdna.rca.ranker import RootCauseRanker

BASE = {
    "latency_p95": 200.0, "self_latency_p95": 60.0, "db_latency_p95": 20.0,
    "error_rate": 0.002, "timeout_rate": 0.0, "slow_trace_participation": 0.1,
    "unique_error_types": 0.0, "log_error_count": 0.0,
}


def _scenario(onsets, peaks, during):
    """Build a store + incident where `during` overrides each service's
    features for the incident window."""
    store = FeatureStore()
    for service in topo.SERVICES:
        for i in range(30):
            store.add(ServiceWindow(i, i * WINDOW_SECONDS, service, dict(BASE)))
        for i in range(30, 40):
            store.add(ServiceWindow(i, i * WINDOW_SECONDS, service,
                                    dict(BASE, **during.get(service, {}))))
    incident = Incident("inc_1", 30, 30 * WINDOW_SECONDS, 31, 31 * WINDOW_SECONDS)
    incident.end_window = 40
    for service, onset in onsets.items():
        incident.services[service] = ServiceIncidentState(
            service=service, onset_window=onset, onset_time=onset * WINDOW_SECONDS,
            peak_score=peaks[service], peak_window=onset,
            last_anomalous_window=39, anomalous_windows=8, total_windows=10,
            peak_z={"latency_p95": 8.0},
        )
    return IncidentContext(incident=incident, store=store, window_seconds=WINDOW_SECONDS)


def _db_incident():
    """Postgres slows first; order and gateway inherit the wait. The gateway
    is the *loudest*, which is what makes strength-ranking fail."""
    return _scenario(
        onsets={topo.POSTGRES: 30, topo.ORDER_SERVICE: 32, topo.API_GATEWAY: 34},
        peaks={topo.POSTGRES: 0.72, topo.ORDER_SERVICE: 0.88, topo.API_GATEWAY: 0.95},
        during={
            topo.POSTGRES: {"latency_p95": 1800.0, "self_latency_p95": 1800.0,
                            "db_latency_p95": 1800.0, "slow_trace_participation": 0.9},
            topo.ORDER_SERVICE: {"latency_p95": 2000.0, "self_latency_p95": 70.0,
                                 "error_rate": 0.05, "timeout_rate": 0.04},
            topo.API_GATEWAY: {"latency_p95": 2100.0, "self_latency_p95": 15.0,
                               "error_rate": 0.12, "timeout_rate": 0.10},
        },
    )


def _rows(ctx):
    return {r["_service"]: r for r in build_candidates(ctx)}


def test_self_latency_share_separates_cause_from_symptom():
    rows = _rows(_db_incident())
    assert rows[topo.POSTGRES]["self_latency_share"] > 0.9
    assert rows[topo.API_GATEWAY]["self_latency_share"] < 0.1


def test_early_onset_favours_the_service_that_moved_first():
    rows = _rows(_db_incident())
    assert rows[topo.POSTGRES]["early_onset"] == 1.0
    assert rows[topo.API_GATEWAY]["early_onset"] == 0.0


def test_propagation_consistency_rewards_the_upstream_cause():
    rows = _rows(_db_incident())
    assert rows[topo.POSTGRES]["propagation_consistency"] > rows[topo.API_GATEWAY]["propagation_consistency"]


def test_dependency_broke_first_penalises_the_symptom():
    rows = _rows(_db_incident())
    assert rows[topo.POSTGRES]["dependency_broke_first"] == 0.0
    assert rows[topo.API_GATEWAY]["dependency_broke_first"] > 0.0


def test_error_originated_is_low_for_inherited_timeouts():
    rows = _rows(_db_incident())
    assert rows[topo.API_GATEWAY]["error_originated"] < 0.3


def test_naive_strength_ranking_picks_the_symptom():
    """The control the project exists to beat: amplification makes the
    gateway look sicker than the database that is breaking it."""
    ctx = _db_incident()
    ranking = RootCauseRanker(AnomalyStrengthScorer()).diagnose(ctx).candidates
    assert ranking[0].service == topo.API_GATEWAY


def test_graph_aware_rankers_pick_the_cause():
    ctx = _db_incident()
    for scorer in (TransparentScorer(), CausalLocalityScorer()):
        top = RootCauseRanker(scorer).diagnose(ctx).candidates[0]
        assert top.service == topo.POSTGRES, f"{scorer.name} blamed {top.service}"


def test_quiet_candidates_do_not_outrank_a_real_deviation():
    """A dependency pulled into the pool that never moved must not win on
    locality features alone."""
    ctx = _db_incident()
    ctx.states[topo.PAYMENT_SERVICE] = ServiceIncidentState(
        service=topo.PAYMENT_SERVICE, onset_window=30, onset_time=30 * WINDOW_SECONDS,
        peak_score=0.05, peak_window=30, last_anomalous_window=30,
        anomalous_windows=0, total_windows=10, peak_z={},
    )
    top = RootCauseRanker(CausalLocalityScorer()).diagnose(ctx).candidates[0]
    assert top.service == topo.POSTGRES


def test_evidence_is_generated_and_quotes_measurements():
    diagnosis = RootCauseRanker(CausalLocalityScorer()).diagnose(_db_incident())
    top = diagnosis.candidates[0]
    assert top.evidence, "a ranked cause must come with evidence"
    joined = " ".join(top.evidence)
    assert "PostgreSQL" in joined
    assert any(ch.isdigit() for ch in joined), "evidence must quote numbers"


def test_contradicting_evidence_is_recorded_for_the_symptom():
    diagnosis = RootCauseRanker(CausalLocalityScorer()).diagnose(_db_incident())
    gateway = next(c for c in diagnosis.candidates if c.service == topo.API_GATEWAY)
    assert gateway.contradicting


def test_confidence_is_a_distribution_over_candidates():
    diagnosis = RootCauseRanker(CausalLocalityScorer()).diagnose(_db_incident())
    total = sum(c.confidence for c in diagnosis.candidates)
    assert total == pytest.approx(1.0, abs=1e-6)
    assert diagnosis.candidates[0].confidence == max(c.confidence for c in diagnosis.candidates)


def test_softmax_confidence_is_not_saturated_for_a_close_call():
    import numpy as np
    conf = softmax_confidence(np.array([0.50, 0.46]))
    assert 0.5 < conf[0] < 0.75, "a near-tie must not be reported as certainty"
