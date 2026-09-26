"""Streaming feature builder: raw telemetry -> per-service window features.

This is the single implementation used by *both* the offline dataset builder
and the live system. It is written as an online operator (ingest / pop_ready)
so the same object works over a replayed file, an in-memory bus, or a Kafka
consumer, and so porting the aggregation to Flink later is a matter of
re-expressing these same reductions.

Aggregation follows how real observability stacks work:
  * counts come from counters  (`requests_total`, `errors_total`, ...)
  * latency percentiles come from sampled duration measurements
  * resource levels come from gauges
  * trace evidence comes from head-sampled traces
"""
from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import numpy as np

from ..config import FEATURE_COLUMNS, WINDOW_SECONDS
from ..telemetry.schema import (
    DeploymentEvent,
    LogRecord,
    MetricPoint,
    SEVERITY_ERROR,
    SEVERITY_WARN,
    Span,
    TelemetryBatch,
    Trace,
)

# Metric names the builder understands.
M_DURATION = "http_request_duration_ms"
M_SELF_DURATION = "self_duration_ms"
M_DEP_DURATION = "dependency_duration_ms"
M_DB_DURATION = "db_query_duration_ms"
M_REQUESTS = "requests_total"
M_ERRORS = "errors_total"
M_TIMEOUTS = "timeouts_total"
M_RETRIES = "retries_total"
M_RESTARTS = "restarts_total"
M_CPU = "cpu_utilization"
M_MEMORY = "memory_mb"
M_CONCURRENT = "concurrent_requests"
M_POOL_UTIL = "pool_utilization"
M_POOL_WAIT = "pool_wait_ms"

_COUNTERS = {M_REQUESTS, M_ERRORS, M_TIMEOUTS, M_RETRIES, M_RESTARTS}
_GAUGES = {M_CPU, M_MEMORY, M_CONCURRENT, M_POOL_UTIL, M_POOL_WAIT}
_DURATIONS = {M_DURATION, M_SELF_DURATION, M_DEP_DURATION, M_DB_DURATION}


def window_index(timestamp: float, window_seconds: int = WINDOW_SECONDS) -> int:
    return int(timestamp // window_seconds)


@dataclass
class ServiceWindow:
    """One row of the feature table: a service observed over one window."""

    window_index: int
    window_start: float
    service: str
    features: Dict[str, float]
    version: str = "v1.0.0"
    run_id: str = ""
    deployed_in_window: bool = False

    def __getitem__(self, key: str) -> float:
        return self.features[key]

    def get(self, key: str, default: float = 0.0) -> float:
        return self.features.get(key, default)

    def vector(self, columns: Optional[List[str]] = None) -> np.ndarray:
        cols = columns or FEATURE_COLUMNS
        return np.array([float(self.features.get(c, 0.0)) for c in cols], dtype=float)

    def to_row(self) -> dict:
        row = {
            "run_id": self.run_id,
            "window_index": self.window_index,
            "window_start": self.window_start,
            "service": self.service,
            "version": self.version,
            "deployed_in_window": self.deployed_in_window,
        }
        row.update({c: float(self.features.get(c, 0.0)) for c in FEATURE_COLUMNS})
        return row


@dataclass
class _Bucket:
    """Mutable accumulator for one (service, window)."""

    durations: List[float] = field(default_factory=list)
    self_durations: List[float] = field(default_factory=list)
    dep_durations: List[float] = field(default_factory=list)
    db_durations: List[float] = field(default_factory=list)
    counters: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    gauges: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    log_errors: int = 0
    log_warns: int = 0
    error_types: set = field(default_factory=set)
    version: str = "v1.0.0"
    deployed: bool = False
    # trace evidence, filled from sampled traces
    slow_traces_seen: int = 0
    slow_traces_dominated: int = 0


def _pct(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, q))


class FeatureBuilder:
    """Bucket telemetry into windows and reduce each bucket to a feature row.

    Usage::

        fb = FeatureBuilder(run_id="inc_0001")
        fb.ingest(batch)
        for window in fb.pop_ready():
            ...
        rows = fb.flush()
    """

    def __init__(
        self,
        run_id: str = "",
        window_seconds: int = WINDOW_SECONDS,
        memory_slope_windows: int = 8,
        trace_baseline_windows: int = 40,
    ) -> None:
        self.run_id = run_id
        self.window_seconds = window_seconds
        self.memory_slope_windows = memory_slope_windows
        self._buckets: Dict[Tuple[int, str], _Bucket] = {}
        self._max_seen_window = -1
        self._memory_history: Dict[str, Deque[Tuple[int, float]]] = defaultdict(
            lambda: deque(maxlen=memory_slope_windows)
        )
        self._trace_durations: Deque[float] = deque(maxlen=trace_baseline_windows * 16)
        self._emitted: set = set()
        #: Traces kept for the UI, newest last.
        self.recent_traces: Deque[Trace] = deque(maxlen=400)

    # -- ingestion ---------------------------------------------------------
    def _bucket(self, widx: int, service: str) -> _Bucket:
        key = (widx, service)
        b = self._buckets.get(key)
        if b is None:
            b = _Bucket()
            self._buckets[key] = b
        return b

    def ingest(self, batch: TelemetryBatch) -> None:
        for m in batch.metrics:
            self._ingest_metric(m)
        for lg in batch.logs:
            self._ingest_log(lg)
        for d in batch.deployments:
            self._ingest_deployment(d)
        if batch.spans:
            for trace in batch.traces():
                self._ingest_trace(trace)

    def _touch(self, widx: int) -> None:
        if widx > self._max_seen_window:
            self._max_seen_window = widx

    def _ingest_metric(self, m: MetricPoint) -> None:
        widx = window_index(m.timestamp, self.window_seconds)
        self._touch(widx)
        b = self._bucket(widx, m.service)
        b.version = m.version
        name = m.metric
        if name == M_DURATION:
            b.durations.append(m.value)
        elif name == M_SELF_DURATION:
            b.self_durations.append(m.value)
        elif name == M_DEP_DURATION:
            b.dep_durations.append(m.value)
        elif name == M_DB_DURATION:
            b.db_durations.append(m.value)
        elif name in _COUNTERS or m.kind == "counter":
            b.counters[name] += m.value
        else:
            b.gauges[name].append(m.value)

    def _ingest_log(self, lg: LogRecord) -> None:
        widx = window_index(lg.timestamp, self.window_seconds)
        self._touch(widx)
        b = self._bucket(widx, lg.service)
        if lg.severity == SEVERITY_ERROR:
            b.log_errors += 1
            b.error_types.add(lg.event)
        elif lg.severity == SEVERITY_WARN:
            b.log_warns += 1

    def _ingest_deployment(self, d: DeploymentEvent) -> None:
        widx = window_index(d.timestamp, self.window_seconds)
        self._touch(widx)
        b = self._bucket(widx, d.service)
        b.deployed = True
        b.version = d.version

    def _ingest_trace(self, trace: Trace) -> None:
        widx = window_index(trace.start_time, self.window_seconds)
        self._touch(widx)
        self.recent_traces.append(trace)
        dur = trace.duration_ms
        threshold = self._slow_trace_threshold()
        self._trace_durations.append(dur)
        if threshold is None or dur < threshold:
            return
        dominant = trace.dominant_service()
        touched = {s.service for s in trace.spans}
        for service in touched:
            b = self._bucket(widx, service)
            b.slow_traces_seen += 1
            if service == dominant:
                b.slow_traces_dominated += 1

    def _slow_trace_threshold(self) -> Optional[float]:
        """A trace is 'slow' relative to recent history, not an absolute ms
        value, so the evidence feature survives load changes."""
        if len(self._trace_durations) < 30:
            return None
        arr = np.fromiter(self._trace_durations, dtype=float)
        return float(max(np.percentile(arr, 90), np.median(arr) * 1.5))

    # -- reduction ---------------------------------------------------------
    def _reduce(self, widx: int, service: str, b: _Bucket) -> ServiceWindow:
        requests = b.counters.get(M_REQUESTS, 0.0)
        errors = b.counters.get(M_ERRORS, 0.0)
        timeouts = b.counters.get(M_TIMEOUTS, 0.0)
        retries = b.counters.get(M_RETRIES, 0.0)
        denom = max(requests, 1.0)

        gauge_mean = lambda n: float(np.mean(b.gauges[n])) if b.gauges.get(n) else 0.0  # noqa: E731
        gauge_max = lambda n: float(np.max(b.gauges[n])) if b.gauges.get(n) else 0.0    # noqa: E731

        memory = gauge_mean(M_MEMORY)
        hist = self._memory_history[service]
        hist.append((widx, memory))
        growth = self._memory_slope(hist)

        slow_seen = b.slow_traces_seen
        participation = (b.slow_traces_dominated / slow_seen) if slow_seen else 0.0

        features = {
            "request_count": requests,
            "concurrent_requests": gauge_mean(M_CONCURRENT),
            "error_rate": errors / denom,
            "timeout_rate": timeouts / denom,
            "retry_rate": retries / denom,
            "restart_count": b.counters.get(M_RESTARTS, 0.0),
            "latency_p50": _pct(b.durations, 50),
            "latency_p95": _pct(b.durations, 95),
            "latency_p99": _pct(b.durations, 99),
            "self_latency_p95": _pct(b.self_durations, 95),
            "dep_latency_p95": _pct(b.dep_durations, 95),
            "db_latency_p95": _pct(b.db_durations, 95),
            "cpu_mean": gauge_mean(M_CPU),
            "cpu_max": gauge_max(M_CPU),
            "memory_mb": memory,
            "memory_growth_rate": growth,
            "pool_utilization": gauge_mean(M_POOL_UTIL),
            "pool_wait_ms": gauge_mean(M_POOL_WAIT),
            "log_error_count": float(b.log_errors),
            "log_warn_count": float(b.log_warns),
            "unique_error_types": float(len(b.error_types)),
            "slow_trace_participation": participation,
        }
        return ServiceWindow(
            window_index=widx,
            window_start=widx * self.window_seconds,
            service=service,
            features=features,
            version=b.version,
            run_id=self.run_id,
            deployed_in_window=b.deployed,
        )

    def _memory_slope(self, hist: Deque[Tuple[int, float]]) -> float:
        """MB per minute, least squares over the retained history."""
        if len(hist) < 3:
            return 0.0
        xs = np.array([h[0] for h in hist], dtype=float)
        ys = np.array([h[1] for h in hist], dtype=float)
        if np.allclose(ys, ys[0]):
            return 0.0
        slope_per_window = float(np.polyfit(xs, ys, 1)[0])
        return slope_per_window * (60.0 / self.window_seconds)

    # -- emission ----------------------------------------------------------
    def pop_ready(self) -> List[ServiceWindow]:
        """Emit every window strictly older than the newest one seen, so a
        window is only reduced once all of its telemetry has arrived."""
        return self._emit(up_to_exclusive=self._max_seen_window)

    def flush(self) -> List[ServiceWindow]:
        """Emit everything, including the window still in progress."""
        return self._emit(up_to_exclusive=self._max_seen_window + 1)

    def _emit(self, up_to_exclusive: int) -> List[ServiceWindow]:
        ready = sorted(
            (k for k in self._buckets if k[0] < up_to_exclusive and k not in self._emitted),
            key=lambda k: (k[0], k[1]),
        )
        out: List[ServiceWindow] = []
        for key in ready:
            widx, service = key
            out.append(self._reduce(widx, service, self._buckets.pop(key)))
            self._emitted.add(key)
        return out


def windows_to_frame(windows: Iterable[ServiceWindow]):
    import pandas as pd

    return pd.DataFrame([w.to_row() for w in windows])
