"""The loop the whole project is: inject -> observe -> detect -> rank -> explain.

These are the tests that would catch a regression anywhere in the chain, so
they run the real simulator through the real pipeline with no stubbing.
"""
import pytest

from conftest import run_fault
from incidentdna import topology as topo
from incidentdna.config import WINDOW_SECONDS
from incidentdna.sim import faults as fault_lib

ALL_FAULTS = sorted(fault_lib.FAULT_NAMES)


@pytest.mark.parametrize("fault_type", ALL_FAULTS)
def test_every_fault_type_is_detected_and_correctly_diagnosed(fault_type):
    analyzer, fault, _ = run_fault(fault_type, seed=11, severity=0.85)
    assert analyzer.incidents, f"{fault_type} produced no incident"
    incident = analyzer.incidents[0]
    assert incident.diagnosis is not None
    assert incident.diagnosis.root_cause == fault.origin, (
        f"{fault_type}: blamed {incident.diagnosis.root_cause}, "
        f"truth was {fault.origin}"
    )


@pytest.mark.parametrize("fault_type", ALL_FAULTS)
def test_detection_is_prompt(fault_type):
    analyzer, fault, engine = run_fault(fault_type, seed=11, severity=0.85)
    incident = analyzer.incidents[0]
    injected_at = engine.start_window + fault.start_window
    delay = (incident.detected_window - injected_at) * WINDOW_SECONDS
    assert 0 <= delay <= 150, f"{fault_type} took {delay}s to detect"


def test_a_clean_run_raises_no_incident():
    analyzer, _, _ = run_fault(None, seed=5)
    assert analyzer.incidents == []


def test_a_traffic_spike_alone_is_not_an_incident():
    """The confounder that load adjustment exists to handle."""
    from incidentdna.config import SimConfig
    from incidentdna.pipeline.analyzer import Analyzer
    from incidentdna.sim.engine import SimulationEngine

    cfg = SimConfig(run_windows=96)
    engine = SimulationEngine(seed=3, config=cfg, run_id="spike")
    engine.schedule_traffic_spike(start_window=45, duration_windows=20, magnitude=2.4)
    analyzer = Analyzer(run_id="spike")
    for _ in range(cfg.run_windows):
        analyzer.ingest(engine.step())
    analyzer.finish()
    assert not analyzer.incidents, "a benign load surge must not page anyone"


def test_the_database_incident_reads_like_the_worked_example():
    """Section 1 of the design: symptom at the gateway, cause at the database,
    propagation in dependency order."""
    analyzer, fault, _ = run_fault("database_latency", seed=3, severity=0.9)
    incident = analyzer.incidents[0]
    states = incident.services

    assert incident.diagnosis.root_cause == topo.POSTGRES
    assert topo.POSTGRES in states and topo.API_GATEWAY in states
    assert states[topo.POSTGRES].onset_window < states[topo.API_GATEWAY].onset_window
    assert incident.symptom is not None and "Gateway" in incident.symptom.describe()

    summary = incident.summary_text(WINDOW_SECONDS)
    for section in ("Severity:", "Detected symptom:", "Most likely root cause:", "Evidence:"):
        assert section in summary


def test_the_incident_closes_after_the_fault_ends():
    analyzer, _, _ = run_fault("database_latency", seed=3, severity=0.9,
                               start_window=30, duration_windows=24)
    incident = analyzer.incidents[0]
    assert incident.status == "resolved"
    assert incident.end_window is not None


def test_the_timeline_records_the_investigation():
    analyzer, _, _ = run_fault("database_latency", seed=3, severity=0.9)
    kinds = {e.kind for e in analyzer.incidents[0].timeline}
    assert {"anomaly", "detection", "diagnosis"} <= kinds


def test_every_ranker_agrees_on_an_unambiguous_incident():
    from incidentdna.rca.baseline import CausalLocalityScorer, TransparentScorer
    from incidentdna.rca.ranker import RootCauseRanker

    analyzer, fault, _ = run_fault("database_latency", seed=3, severity=0.9)
    incident = analyzer.incidents[0]
    ctx = analyzer.manager.context(incident)
    for scorer in (TransparentScorer(), CausalLocalityScorer()):
        assert RootCauseRanker(scorer).diagnose(ctx).candidates[0].service == fault.origin


def test_candidate_pool_includes_quiet_dependencies():
    """A cause can be subtler than its symptom, so dependencies of an
    abnormal service are candidates even when they stayed under threshold."""
    analyzer, _, _ = run_fault("payment_timeout", seed=11, severity=0.85)
    incident = analyzer.incidents[0]
    ctx = analyzer.manager.context(incident)
    assert set(ctx.states) >= set(ctx.confirmed)
    assert len(ctx.states) > len(ctx.confirmed)


def test_feature_rows_survive_a_round_trip_through_the_dataset_format():
    import pandas as pd

    from incidentdna.sim.runs import RunSpec, execute_run, frame_to_windows

    result = execute_run(RunSpec(run_id="rt", ordinal=0, seed=1, fault_type="cpu_saturation",
                                 origin=topo.ORDER_SERVICE, severity=0.8))
    frame = pd.DataFrame(result.rows())
    restored = frame_to_windows(frame, "rt")
    assert len(restored) == len(result.windows)
    a, b = result.windows[10], restored[10]
    assert a.service == b.service and a.window_index == b.window_index
    assert a.features == pytest.approx(b.features)
