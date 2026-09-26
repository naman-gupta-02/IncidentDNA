"""Stage one: does it fire when it should, and stay quiet when it shouldn't."""
import numpy as np
import pytest

from incidentdna.config import WINDOW_SECONDS
from incidentdna.detect.base import observe
from incidentdna.detect.detectors import RollingZScoreDetector, StaticThresholdDetector
from incidentdna.detect.ensemble import EnsembleDetector
from incidentdna.pipeline.features import ServiceWindow
from incidentdna.pipeline.store import FeatureStore

BASE = {
    "latency_p95": 200.0, "latency_p99": 260.0, "self_latency_p95": 60.0,
    "db_latency_p95": 20.0, "error_rate": 0.002, "timeout_rate": 0.0,
    "cpu_mean": 0.3, "memory_growth_rate": 0.0, "pool_utilization": 0.35,
    "log_error_count": 0.0, "restart_count": 0.0,
    "request_count": 600.0, "concurrent_requests": 5.0,
}


def _store(n=24, jitter=0.02, seed=0):
    rng = np.random.default_rng(seed)
    store = FeatureStore()
    for i in range(n):
        feats = {k: float(v * rng.normal(1.0, jitter)) for k, v in BASE.items()}
        store.add(ServiceWindow(i, i * WINDOW_SECONDS, "svc", feats))
    return store


def _window(idx, **overrides):
    feats = dict(BASE)
    feats.update(overrides)
    return ServiceWindow(idx, idx * WINDOW_SECONDS, "svc", feats)


def test_quiet_when_nothing_changed():
    store = _store()
    det = RollingZScoreDetector()
    assert det.score(observe(_window(24), store)) < 0.3


def test_fires_on_a_latency_step_change():
    store = _store()
    det = RollingZScoreDetector()
    obs = observe(_window(24, latency_p95=2400.0, latency_p99=3000.0, self_latency_p95=1900.0), store)
    assert det.score(obs) > 0.8


def test_load_adjustment_suppresses_a_pure_traffic_spike():
    """Latency rising in step with a 3x traffic rise is queueing, not a fault."""
    store = _store()
    spike = _window(24, request_count=1800.0, concurrent_requests=15.0,
                    latency_p95=340.0, latency_p99=430.0, cpu_mean=0.52)
    obs = observe(spike, store)
    naive = RollingZScoreDetector(load_alpha=0.0).score(obs)
    adjusted = RollingZScoreDetector(load_alpha=0.6).score(obs)
    assert naive > adjusted, "load adjustment must reduce the score"
    assert adjusted < naive * 0.85


def test_load_adjustment_still_reports_a_genuine_fault_under_load():
    store = _store()
    real = _window(24, request_count=1800.0, concurrent_requests=15.0,
                   latency_p95=4000.0, self_latency_p95=3600.0, error_rate=0.22)
    assert RollingZScoreDetector(load_alpha=0.6).score(observe(real, store)) > 0.8


def test_abstains_without_enough_history():
    store = _store(n=3)
    assert RollingZScoreDetector().score(observe(_window(3, latency_p95=9000.0), store)) is None


def test_constant_baseline_does_not_produce_an_infinite_z():
    """error_rate is often exactly 0 for many windows; a rounding-level move
    must not read as an unbounded deviation."""
    store = FeatureStore()
    for i in range(20):
        feats = dict(BASE, error_rate=0.0)
        store.add(ServiceWindow(i, i * WINDOW_SECONDS, "svc", feats))
    obs = observe(_window(20, error_rate=0.0005), store)
    assert obs.z["error_rate"] < 1.0


def test_static_threshold_reports_which_rules_broke():
    det = StaticThresholdDetector()
    obs = observe(_window(0, error_rate=0.3, pool_utilization=0.99), FeatureStore())
    assert set(det.breached(obs)) == {"error_rate", "pool_utilization"}


def test_ensemble_renormalises_over_available_detectors():
    """Unfitted learned detectors abstain rather than scoring zero."""
    store = _store()
    ens = EnsembleDetector.full()          # IF and AE are not fitted
    anomaly = ens.score_window(_window(24, latency_p95=2400.0, self_latency_p95=1900.0), store)
    assert set(anomaly.method_scores) == {"static_threshold", "rolling_zscore"}
    assert anomaly.score > 0.5, "abstaining detectors must not drag the score down"


def test_scoring_a_group_matches_scoring_one_at_a_time():
    store = _store()
    for svc in ("a", "b"):
        for i in range(24):
            store.add(ServiceWindow(i, i * WINDOW_SECONDS, svc, dict(BASE)))
    ens = EnsembleDetector.statistical_only()
    w = [ServiceWindow(24, 24 * WINDOW_SECONDS, s, dict(BASE, latency_p95=2000.0)) for s in ("a", "b")]
    grouped = ens.score_group(w, store)
    singles = [ens.score_window(x, store) for x in w]
    assert [g.score for g in grouped] == pytest.approx([s.score for s in singles])
