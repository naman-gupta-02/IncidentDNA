"""The simulator has to be right, or every measured number is meaningless."""
import numpy as np
import pytest

from conftest import run_fault
from incidentdna import topology as topo
from incidentdna.config import SimConfig
from incidentdna.sim import faults as fault_lib
from incidentdna.sim.engine import SimulationEngine
from incidentdna.sim.runs import build_matrix, execute_run


def _run(fault_type=None, seed=2, severity=0.85, origin=None, windows=96):
    cfg = SimConfig(run_windows=windows)
    engine = SimulationEngine(seed=seed, config=cfg, run_id="s")
    fault = None
    if fault_type:
        fault = engine.inject(fault_type, origin=origin, severity=severity,
                              start_window=45, duration_windows=32)
    for _ in range(windows):
        engine.step()
    return engine, fault


def _latency(engine, service):
    return np.array([snap[service].total_latency_ms for snap in engine.snapshots])


def test_a_clean_run_is_stable():
    engine, _ = _run()
    for service in topo.SERVICES:
        lat = _latency(engine, service)
        assert lat.max() < lat[20:40].mean() * 4, f"{service} drifted with no fault"


def test_a_database_fault_propagates_upstream_in_dependency_order():
    engine, fault = _run("database_latency", severity=0.9)

    def onset(service):
        lat = _latency(engine, service)
        base = np.median(lat[25:44])
        above = np.where(lat[44:] > base * 1.8)[0]
        return 44 + above[0] if len(above) else None

    pg, order, gw = onset(topo.POSTGRES), onset(topo.ORDER_SERVICE), onset(topo.API_GATEWAY)
    assert pg is not None and order is not None and gw is not None
    assert pg <= order <= gw, f"expected postgres -> order -> gateway, got {pg}, {order}, {gw}"


def test_a_fault_does_not_disturb_an_unrelated_branch():
    engine, _ = _run("database_latency", severity=0.9)
    payment = _latency(engine, topo.PAYMENT_SERVICE)
    assert payment[45:77].mean() < payment[20:44].mean() * 1.5


@pytest.mark.parametrize("fault_type", sorted(fault_lib.FAULT_NAMES))
def test_every_fault_moves_its_origin(fault_type):
    engine, fault = _run(fault_type, severity=0.9)
    assert fault.origin in engine.affected_services(fault)


def test_amplification_can_make_a_symptom_louder_than_its_cause():
    """The reason 'rank by anomaly strength' is not good enough: once a
    dependency crosses a client timeout the caller pays timeout x attempts."""
    engine, _ = _run("payment_timeout", severity=1.0)
    order_errors = max(s[topo.ORDER_SERVICE].error_rate for s in engine.snapshots)
    payment_errors = max(s[topo.PAYMENT_SERVICE].error_rate for s in engine.snapshots)
    assert order_errors > payment_errors


def test_propagation_lag_is_fixed_per_run_and_in_range():
    engine, _ = _run()
    for (caller, callee), lag in engine.edge_lag.items():
        assert 10.0 <= lag <= 45.0


def test_traffic_spikes_raise_load_without_saturating_a_healthy_service():
    cfg = SimConfig(run_windows=96)
    engine = SimulationEngine(seed=4, config=cfg, run_id="spike")
    engine.schedule_traffic_spike(start_window=40, duration_windows=20, magnitude=2.6)
    for _ in range(96):
        engine.step()
    rps = np.array([s[topo.API_GATEWAY].rps for s in engine.snapshots])
    lat = _latency(engine, topo.ORDER_SERVICE)
    assert rps[45:55].mean() > rps[20:35].mean() * 1.6, "the spike must actually arrive"
    assert lat[45:55].mean() < lat[20:35].mean() * 2.0, "but it must be absorbed"


def test_runs_are_deterministic_for_a_seed():
    a, _ = _run("cpu_saturation", seed=9)
    b, _ = _run("cpu_saturation", seed=9)
    assert _latency(a, topo.ORDER_SERVICE).tolist() == _latency(b, topo.ORDER_SERVICE).tolist()


def test_runs_occupy_disjoint_window_ranges():
    specs = build_matrix(normal_runs=2, runs_per_fault=1, seed=1)
    seen = set()
    for spec in specs[:6]:
        result = execute_run(spec)
        windows = {w.window_index for w in result.windows}
        assert not (windows & seen), "runs must not share absolute window indices"
        seen |= windows


def test_the_experiment_matrix_is_stratified_by_fault_type():
    specs = build_matrix(normal_runs=30, runs_per_fault=20, seed=1)
    for fault_type in [None] + list(fault_lib.FAULT_NAMES):
        group = [s for s in specs if s.fault_type == fault_type]
        splits = {s.split for s in group}
        assert splits == {"train", "val", "test"}, f"{fault_type} missing a split"


def test_ground_truth_records_origin_and_affected_services():
    specs = build_matrix(normal_runs=0, runs_per_fault=1, seed=3,
                         fault_types=["database_latency"])
    result = execute_run(specs[0])
    gt = result.ground_truth
    assert gt is not None
    assert gt.root_cause_service == topo.POSTGRES
    assert gt.root_cause_service in gt.affected_services
    assert gt.end_window > gt.start_window
