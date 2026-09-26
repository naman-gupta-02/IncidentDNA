"""Feature builder and store: the numbers everything else is derived from."""
import numpy as np
import pytest

from incidentdna.config import FEATURE_COLUMNS, WINDOW_SECONDS
from incidentdna.pipeline import features as F
from incidentdna.pipeline.features import FeatureBuilder, ServiceWindow, window_index
from incidentdna.pipeline.store import FeatureStore
from incidentdna.telemetry.schema import LogRecord, MetricPoint, TelemetryBatch

T0 = 1_700_000_000.0


def _batch(service="order-service", t0=T0, durations=(10.0,), requests=100.0, errors=5.0):
    b = TelemetryBatch()
    for i, d in enumerate(durations):
        b.metrics.append(MetricPoint(t0 + i * 0.1, service, F.M_DURATION, d, "histogram"))
    b.metrics.append(MetricPoint(t0, service, F.M_REQUESTS, requests, "counter"))
    b.metrics.append(MetricPoint(t0, service, F.M_ERRORS, errors, "counter"))
    b.metrics.append(MetricPoint(t0, service, F.M_CPU, 0.4))
    b.metrics.append(MetricPoint(t0, service, F.M_MEMORY, 300.0))
    return b


def test_window_index_buckets_by_wall_clock():
    assert window_index(T0) == int(T0 // WINDOW_SECONDS)
    assert window_index(T0 + WINDOW_SECONDS) == window_index(T0) + 1


def test_reduction_produces_every_declared_feature():
    fb = FeatureBuilder()
    fb.ingest(_batch())
    rows = fb.flush()
    assert len(rows) == 1
    assert set(rows[0].features) == set(FEATURE_COLUMNS)


def test_counters_sum_and_rates_divide_by_requests():
    fb = FeatureBuilder()
    fb.ingest(_batch(requests=200.0, errors=20.0))
    w = fb.flush()[0]
    assert w["request_count"] == 200.0
    assert w["error_rate"] == pytest.approx(0.1)


def test_percentiles_come_from_the_sampled_durations():
    fb = FeatureBuilder()
    fb.ingest(_batch(durations=tuple(float(x) for x in range(1, 101))))
    w = fb.flush()[0]
    assert w["latency_p50"] == pytest.approx(50.5, abs=1.0)
    assert w["latency_p95"] == pytest.approx(95.05, abs=1.0)


def test_pop_ready_withholds_the_window_still_being_filled():
    fb = FeatureBuilder()
    fb.ingest(_batch(t0=T0))
    assert fb.pop_ready() == []          # only one window seen so far
    fb.ingest(_batch(t0=T0 + WINDOW_SECONDS))
    ready = fb.pop_ready()
    assert len(ready) == 1 and ready[0].window_index == window_index(T0)


def test_logs_are_counted_by_severity_and_unique_event():
    fb = FeatureBuilder()
    b = _batch()
    for event in ("database_timeout", "database_timeout", "oom_killed"):
        b.logs.append(LogRecord(T0, "order-service", "ERROR", event))
    b.logs.append(LogRecord(T0, "order-service", "WARN", "slow_query"))
    fb.ingest(b)
    w = fb.flush()[0]
    assert w["log_error_count"] == 3
    assert w["log_warn_count"] == 1
    assert w["unique_error_types"] == 2


def test_memory_growth_rate_is_reported_per_minute():
    fb = FeatureBuilder()
    for i in range(6):
        b = _batch(t0=T0 + i * WINDOW_SECONDS)
        for m in b.metrics:
            if m.metric == F.M_MEMORY:
                m.value = 300.0 + 10.0 * i     # 10 MB per 15s window
        fb.ingest(b)
    rows = fb.flush()
    # 10 MB/window at 15s windows is 40 MB/min.
    assert rows[-1]["memory_growth_rate"] == pytest.approx(40.0, rel=0.02)


# --- store ----------------------------------------------------------------
def _window(idx, value, service="s", metric="latency_p95"):
    return ServiceWindow(idx, idx * WINDOW_SECONDS, service, {metric: value})


def test_baseline_excludes_the_window_being_judged():
    store = FeatureStore()
    for i in range(20):
        store.add(_window(i, 100.0))
    store.add(_window(20, 9999.0))
    median, _, n = store.robust_baseline("s", "latency_p95", 20)
    assert median == 100.0, "the anomalous window must not inflate its own baseline"
    assert n == 20


def test_robust_baseline_resists_contamination():
    store = FeatureStore()
    for i in range(20):
        store.add(_window(i, 100.0 if i < 17 else 5000.0))
    median, _, _ = store.robust_baseline("s", "latency_p95", 20)
    mean, _, _ = store.baseline("s", "latency_p95", 20)
    assert median == 100.0
    assert mean > 700.0, "the mean is the thing the median is protecting us from"


def test_baseline_is_none_until_there_is_enough_history():
    store = FeatureStore()
    for i in range(4):
        store.add(_window(i, 100.0))
    assert store.robust_baseline("s", "latency_p95", 4) is None


def test_window_slice_is_inclusive_on_both_ends():
    store = FeatureStore()
    for i in range(10):
        store.add(_window(i, float(i)))
    got = store.window_slice("s", 3, 6)
    assert [w.window_index for w in got] == [3, 4, 5, 6]
