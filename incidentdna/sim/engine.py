"""Generative model of the instrumented e-commerce system.

The engine advances the world one window at a time and emits real telemetry
records. It models three things that make root-cause analysis hard:

1. **Propagation with lag.** A caller observes its callee's latency delayed by
   a per-edge lag (10-45s), so anomaly onsets are ordered but noisily.
2. **Amplification.** Once a dependency crosses a client timeout, the caller
   pays `timeout * attempts` and starts erroring. A symptom can therefore look
   *more* severe than its cause, which defeats "rank by anomaly strength".
3. **Confounders.** Traffic waves and spikes slow everything down at once with
   no root cause at all, which is what generates false-positive pressure.

The simulator only ever labels the *origin* of a fault. Which services end up
affected, and in what order, is emergent.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .. import topology as topo
from ..config import SimConfig
from ..pipeline.features import (
    M_CONCURRENT,
    M_CPU,
    M_DB_DURATION,
    M_DEP_DURATION,
    M_DURATION,
    M_ERRORS,
    M_MEMORY,
    M_POOL_UTIL,
    M_POOL_WAIT,
    M_REQUESTS,
    M_RESTARTS,
    M_RETRIES,
    M_SELF_DURATION,
    M_TIMEOUTS,
)
from ..telemetry.schema import (
    DeploymentEvent,
    LogRecord,
    MetricPoint,
    SEVERITY_ERROR,
    SEVERITY_INFO,
    SEVERITY_WARN,
    SPAN_KIND_CLIENT,
    SPAN_KIND_SERVER,
    Span,
    TelemetryBatch,
    new_id,
)
from . import faults as fault_lib

# Duration measurements emitted per service per window. Real deployments
# export a histogram; we export sampled exemplars, which is equivalent for
# percentile estimation and keeps the record count tractable.
DURATION_SAMPLES = 40

#: Baseline behaviour of each service when nothing is wrong.
#
#: Capacities give every service roughly 6-8x headroom over its baseline rate.
#: That is deliberate and it matters for the labels: a service provisioned to
#: run at 40% utilisation genuinely saturates under a 2.4x traffic surge, and
#: the resulting latency explosion is a real capacity incident — but the
#: experiment matrix labels those runs "normal". Rather than teach the
#: detector to ignore true degradation, the modelled system is provisioned the
#: way a real one is, so an ordinary load surge is absorbed and only an
#: injected fault (`cpu_saturation` cuts capacity directly) drives a service
#: into saturation.
SERVICE_PROFILES: Dict[str, dict] = {
    topo.API_GATEWAY: dict(self_ms=12.0, sigma=0.32, capacity_rps=600.0, cpu=0.16, memory_mb=220.0),
    topo.ORDER_SERVICE: dict(self_ms=26.0, sigma=0.38, capacity_rps=260.0, cpu=0.24, memory_mb=310.0),
    topo.PAYMENT_SERVICE: dict(self_ms=145.0, sigma=0.42, capacity_rps=340.0, cpu=0.19, memory_mb=185.0),
    topo.INVENTORY_SERVICE: dict(self_ms=17.0, sigma=0.34, capacity_rps=560.0, cpu=0.15, memory_mb=205.0),
    topo.POSTGRES: dict(self_ms=9.0, sigma=0.52, capacity_rps=900.0, cpu=0.21, memory_mb=512.0),
}

BASE_ERROR_RATE = 0.0015

#: Which log event a caller writes when a given dependency fails it.
_DEP_FAILURE_EVENT = {
    topo.POSTGRES: "database_timeout",
    topo.PAYMENT_SERVICE: "payment_timeout",
    topo.INVENTORY_SERVICE: "inventory_unavailable",
    topo.ORDER_SERVICE: "upstream_timeout",
}


@dataclass
class ActiveFault:
    """A fault currently being injected."""

    fault_id: str
    spec: fault_lib.FaultSpec
    origin: str
    severity: float
    start_window: int
    duration_windows: int
    state: Dict[str, float] = field(default_factory=dict)
    deployed: bool = False

    def progress(self, window: int) -> float:
        if self.duration_windows <= 0:
            return 1.0
        return (window - self.start_window) / float(self.duration_windows)

    def active(self, window: int) -> bool:
        return self.start_window <= window < self.start_window + self.duration_windows


@dataclass
class TrafficSpike:
    """A benign load surge: a confounder with no root cause."""

    start_window: int
    duration_windows: int
    magnitude: float

    def factor(self, window: int) -> float:
        if not (self.start_window <= window < self.start_window + self.duration_windows):
            return 1.0
        p = (window - self.start_window) / float(self.duration_windows)
        shape = math.sin(math.pi * min(1.0, max(0.0, p)))
        return 1.0 + (self.magnitude - 1.0) * shape


@dataclass
class ServiceSnapshot:
    """Everything the engine computed about one service in one window."""

    service: str
    rps: float
    self_latency_ms: float
    total_latency_ms: float
    dep_latency_ms: float
    db_latency_ms: float
    sigma: float
    error_rate: float
    timeout_rate: float
    retry_rate: float
    cpu: float
    memory_mb: float
    pool_utilization: float
    pool_wait_ms: float
    restarts: int
    version: str
    dominant_failing_dep: Optional[str] = None
    edge_fail: Dict[str, float] = field(default_factory=dict)


def _lognormal_samples(rng: np.random.Generator, median: float, sigma: float, n: int) -> np.ndarray:
    median = max(median, 0.05)
    sigma = float(np.clip(sigma, 0.05, 1.6))
    return rng.lognormal(mean=math.log(median), sigma=sigma, size=n)


def _p_exceed(median: float, sigma: float, threshold: float) -> float:
    """P(X > threshold) for X ~ lognormal(median, sigma)."""
    if median <= 0:
        return 0.0
    if threshold <= 0:
        return 1.0
    sigma = max(sigma, 1e-3)
    z = (math.log(threshold) - math.log(median)) / sigma
    return 0.5 * math.erfc(z / math.sqrt(2.0))


class SimulationEngine:
    """Advance the modelled system window by window, emitting telemetry."""

    def __init__(
        self,
        seed: int = 0,
        config: Optional[SimConfig] = None,
        start_time: Optional[float] = None,
        run_id: str = "run",
        base_rps: Optional[float] = None,
        emit_traces: bool = True,
    ) -> None:
        self.cfg = config or SimConfig()
        self.rng = np.random.default_rng(seed)
        self.run_id = run_id
        self.window_seconds = self.cfg.window_seconds
        self.emit_traces = emit_traces
        base = start_time if start_time is not None else 1_700_000_000.0
        # Snap to a window boundary so window indices line up with the
        # feature builder's `floor(ts / window_seconds)`.
        self.start_time = float(int(base // self.window_seconds) * self.window_seconds)
        self.start_window = int(self.start_time // self.window_seconds)
        self.window = 0
        self.base_rps = base_rps if base_rps is not None else self.cfg.base_rps

        self.active_faults: List[ActiveFault] = []
        self.completed_faults: List[ActiveFault] = []
        self.spikes: List[TrafficSpike] = []
        self.deployments: List[DeploymentEvent] = []
        self.versions: Dict[str, str] = {s: "v1.4.0" for s in topo.SERVICES}

        # Per-edge propagation lag in seconds, fixed for the life of the run.
        self.edge_lag: Dict[Tuple[str, str], float] = {
            (e.caller, e.callee): float(self.rng.uniform(10.0, 45.0)) for e in topo.EDGES
        }
        # History of modelled medians, indexed by window.
        self._lat_hist: Dict[str, List[float]] = {s: [] for s in topo.SERVICES}
        self._err_hist: Dict[str, List[float]] = {s: [] for s in topo.SERVICES}
        self.snapshots: List[Dict[str, ServiceSnapshot]] = []
        self._traffic_phase = float(self.rng.uniform(0, 2 * math.pi))
        self._pending_deploys: List[Tuple[int, str, str]] = []

    # -- scheduling --------------------------------------------------------
    def inject(
        self,
        fault_name: str,
        origin: Optional[str] = None,
        severity: Optional[float] = None,
        duration_windows: Optional[int] = None,
        start_window: Optional[int] = None,
    ) -> ActiveFault:
        spec = fault_lib.get(fault_name)
        if origin is None:
            origin = str(self.rng.choice(spec.origin_candidates))
        elif origin not in spec.origin_candidates:
            raise ValueError(f"{fault_name} cannot originate in {origin}")
        if severity is None:
            severity = float(self.rng.uniform(0.35, 1.0))
        if duration_windows is None:
            lo, hi = self.cfg.fault_duration_windows
            duration_windows = int(self.rng.integers(lo, hi + 1))
        af = ActiveFault(
            fault_id=f"flt_{new_id(8)}",
            spec=spec,
            origin=origin,
            severity=float(severity),
            start_window=self.window if start_window is None else start_window,
            duration_windows=duration_windows,
        )
        self.active_faults.append(af)
        return af

    def schedule_traffic_spike(
        self, start_window: int, duration_windows: int, magnitude: float
    ) -> None:
        self.spikes.append(TrafficSpike(start_window, duration_windows, magnitude))

    def schedule_deployment(self, window: int, service: str, version: str) -> None:
        self._pending_deploys.append((window, service, version))

    # -- time --------------------------------------------------------------
    def window_start_time(self, window: Optional[int] = None) -> float:
        w = self.window if window is None else window
        return self.start_time + w * self.window_seconds

    def absolute_window_index(self, window: Optional[int] = None) -> int:
        return self.start_window + (self.window if window is None else window)

    # -- propagation helpers ----------------------------------------------
    def _lagged(self, hist: List[float], window: int, lag_seconds: float, default: float) -> float:
        """Value of a history series as observed `lag_seconds` ago, with linear
        interpolation between the two bracketing windows."""
        lag_windows = lag_seconds / self.window_seconds
        pos = window - lag_windows
        if pos <= 0:
            return hist[0] if hist else default
        lo = int(math.floor(pos))
        frac = pos - lo
        hi = min(lo + 1, len(hist) - 1)
        lo = min(lo, len(hist) - 1)
        if lo < 0 or not hist:
            return default
        return hist[lo] * (1.0 - frac) + hist[hi] * frac

    def _traffic_factor(self, window: int) -> float:
        t = window / max(1.0, self.cfg.run_windows)
        wave = 1.0 + 0.18 * math.sin(2 * math.pi * t * 1.5 + self._traffic_phase)
        noise = float(self.rng.normal(1.0, 0.035))
        spike = 1.0
        for s in self.spikes:
            spike *= s.factor(window)
        return max(0.25, wave * noise * spike)

    # -- the step ----------------------------------------------------------
    def step(self) -> TelemetryBatch:
        """Advance one window and return everything the system emitted."""
        w = self.window
        batch = TelemetryBatch()
        t0 = self.window_start_time(w)

        for window, service, version in list(self._pending_deploys):
            if window == w:
                self._record_deployment(batch, service, version, t0 + 1.0)
                self._pending_deploys.remove((window, service, version))

        effects = self._collect_fault_effects(w, batch, t0)
        traffic = self._traffic_factor(w)
        rps = self._request_rates(traffic)
        snapshots = self._resolve_window(w, rps, effects)

        for s in topo.SERVICES:
            self._lat_hist[s].append(snapshots[s].total_latency_ms)
            self._err_hist[s].append(snapshots[s].error_rate)
        self.snapshots.append(snapshots)

        self._emit_metrics(batch, snapshots, t0)
        self._emit_logs(batch, snapshots, effects, t0)
        if self.emit_traces:
            self._emit_traces(batch, snapshots, t0)

        for af in list(self.active_faults):
            if not af.active(w) and w >= af.start_window:
                self.active_faults.remove(af)
                self.completed_faults.append(af)

        self.window += 1
        return batch

    # -- internals ---------------------------------------------------------
    def _collect_fault_effects(
        self, w: int, batch: TelemetryBatch, t0: float
    ) -> Dict[str, fault_lib.FaultEffect]:
        merged: Dict[str, fault_lib.FaultEffect] = {}
        for af in self.active_faults:
            if not af.active(w):
                continue
            ctx = fault_lib.FaultContext(
                progress=af.progress(w),
                elapsed_seconds=(w - af.start_window) * self.window_seconds,
                severity=af.severity,
                rng=self.rng,
                state=af.state,
            )
            eff = af.spec.effect(ctx)
            if eff.deploy_version and not af.deployed:
                af.deployed = True
                self._record_deployment(batch, af.origin, eff.deploy_version, t0 + 0.5)
            prev = merged.get(af.origin)
            merged[af.origin] = _merge_effects(prev, eff) if prev else eff
        return merged

    def _record_deployment(self, batch: TelemetryBatch, service: str, version: str, ts: float) -> None:
        prev = self.versions.get(service)
        self.versions[service] = version
        ev = DeploymentEvent(timestamp=ts, service=service, version=version, previous_version=prev)
        batch.deployments.append(ev)
        self.deployments.append(ev)

    def _request_rates(self, traffic: float) -> Dict[str, float]:
        rps = {s: 0.0 for s in topo.SERVICES}
        for ep in topo.ENTRYPOINTS:
            rps[ep] += self.base_rps * traffic
        # Callees inherit their callers' rate scaled by fanout; the graph is a
        # DAG so a topological pass is enough.
        for service in reversed(topo.reverse_topological_order()):
            for callee in topo.callees(service):
                rps[callee] += rps[service] * topo.edge(service, callee).fanout
        return rps

    def _resolve_window(
        self, w: int, rps: Dict[str, float], effects: Dict[str, fault_lib.FaultEffect]
    ) -> Dict[str, ServiceSnapshot]:
        snaps: Dict[str, ServiceSnapshot] = {}
        for service in topo.reverse_topological_order():   # callees first
            snaps[service] = self._resolve_service(w, service, rps, effects, snaps)
        return snaps

    def _resolve_service(
        self,
        w: int,
        service: str,
        rps: Dict[str, float],
        effects: Dict[str, fault_lib.FaultEffect],
        snaps: Dict[str, ServiceSnapshot],
    ) -> ServiceSnapshot:
        prof = SERVICE_PROFILES[service]
        eff = effects.get(service, fault_lib.FaultEffect())
        rate = rps[service]

        # Queueing: latency inflates as utilisation approaches capacity.
        capacity = max(1.0, prof["capacity_rps"] * eff.capacity_mult)
        util = min(0.97, rate / capacity)
        base_util = min(0.97, self._baseline_rate(service) / prof["capacity_rps"])
        queue_mult = (1.0 - base_util) / max(0.03, 1.0 - util)

        self_ms = prof["self_ms"] * queue_mult * eff.self_latency_mult + eff.self_latency_add_ms
        self_ms *= float(self.rng.normal(1.0, 0.04))
        self_ms = max(0.5, self_ms)
        sigma = prof["sigma"] + eff.self_latency_sigma_add

        dep_total = 0.0
        db_latency = 0.0
        edge_fail: Dict[str, float] = {}
        timeout_calls = 0.0
        retry_calls = 0.0
        total_calls = 0.0

        for callee in topo.callees(service):
            e = topo.edge(service, callee)
            lag = self.edge_lag[(service, callee)]
            callee_lat = self._lagged(
                self._lat_hist[callee], w, lag, SERVICE_PROFILES[callee]["self_ms"]
            )
            # The callee's *current* window is not in _lat_hist yet; blend it in
            # for the un-lagged fraction so short lags propagate within a window.
            if lag < self.window_seconds:
                frac = 1.0 - lag / self.window_seconds
                callee_lat = callee_lat * (1 - frac) + snaps[callee].total_latency_ms * frac
            callee_err = self._lagged(self._err_hist[callee], w, lag, BASE_ERROR_RATE)
            callee_sigma = snaps[callee].sigma

            p_timeout = _p_exceed(callee_lat, callee_sigma, e.timeout_ms)
            attempts = e.retries + 1
            p_all_timeout = p_timeout ** attempts
            expected_failed_attempts = sum(p_timeout ** k for k in range(1, attempts + 1))
            cost = e.timeout_ms * expected_failed_attempts + (1.0 - p_all_timeout) * min(
                callee_lat, e.timeout_ms
            )
            # Callee-returned errors also fail the call, after the same lag.
            p_edge_fail = 1.0 - (1.0 - p_all_timeout) * (1.0 - callee_err)
            p_edge_fail = 1.0 - (1.0 - p_edge_fail) ** max(1.0, e.fanout)

            dep_total += e.fanout * cost
            if callee in topo.DATABASE_SERVICES:
                db_latency = max(db_latency, cost)
            edge_fail[callee] = p_edge_fail
            total_calls += e.fanout
            timeout_calls += e.fanout * p_all_timeout
            retry_calls += e.fanout * max(0.0, expected_failed_attempts - p_all_timeout)

        total_latency = self_ms + dep_total
        if service in topo.DATABASE_SERVICES:
            db_latency = self_ms

        p_dep_fail = 1.0
        for p in edge_fail.values():
            p_dep_fail *= 1.0 - p
        p_dep_fail = 1.0 - p_dep_fail
        error_rate = min(0.98, 1.0 - (1.0 - p_dep_fail) * (1.0 - BASE_ERROR_RATE) * (
            1.0 - eff.intrinsic_error_rate
        ))

        cpu = float(np.clip(prof["cpu"] + 0.45 * util + eff.cpu_add + self.rng.normal(0, 0.012), 0.01, 1.0))
        memory = prof["memory_mb"] + eff.memory_add_mb + float(self.rng.normal(0, 3.0))

        if eff.pool_utilization is not None:
            pool_util = eff.pool_utilization
        elif service in topo.DATABASE_SERVICES:
            pool_util = float(np.clip(0.30 + 0.55 * util + self.rng.normal(0, 0.02), 0.0, 1.0))
        else:
            pool_util = 0.0
        pool_wait = eff.pool_wait_ms

        dominant = None
        if edge_fail:
            dominant = max(edge_fail.items(), key=lambda kv: kv[1])[0]
            if edge_fail[dominant] < 0.01:
                dominant = None

        return ServiceSnapshot(
            service=service,
            rps=rate,
            self_latency_ms=self_ms,
            total_latency_ms=total_latency,
            dep_latency_ms=dep_total,
            db_latency_ms=db_latency,
            sigma=float(np.clip(sigma, 0.08, 1.5)),
            error_rate=error_rate,
            timeout_rate=min(1.0, timeout_calls / max(total_calls, 1.0)) if total_calls else 0.0,
            retry_rate=(retry_calls / max(total_calls, 1.0)) if total_calls else 0.0,
            cpu=cpu,
            memory_mb=max(20.0, memory),
            pool_utilization=pool_util,
            pool_wait_ms=pool_wait,
            restarts=1 if eff.restart else 0,
            version=self.versions[service],
            dominant_failing_dep=dominant,
            edge_fail=edge_fail,
        )

    def _baseline_rate(self, service: str) -> float:
        """Unperturbed request rate, used to normalise the queueing term."""
        rates = {s: 0.0 for s in topo.SERVICES}
        for ep in topo.ENTRYPOINTS:
            rates[ep] += self.base_rps
        for s in reversed(topo.reverse_topological_order()):
            for callee in topo.callees(s):
                rates[callee] += rates[s] * topo.edge(s, callee).fanout
        return rates[service]

    # -- emission ----------------------------------------------------------
    def _emit_metrics(
        self, batch: TelemetryBatch, snaps: Dict[str, ServiceSnapshot], t0: float
    ) -> None:
        W = self.window_seconds
        rng = self.rng
        for service, snap in snaps.items():
            ts_base = t0
            add = batch.metrics.append

            dur = _lognormal_samples(rng, snap.total_latency_ms, snap.sigma, DURATION_SAMPLES)
            selfd = _lognormal_samples(rng, snap.self_latency_ms, snap.sigma, DURATION_SAMPLES)
            offsets = np.sort(rng.uniform(0, W, DURATION_SAMPLES))
            for i in range(DURATION_SAMPLES):
                ts = ts_base + float(offsets[i])
                add(MetricPoint(ts, service, M_DURATION, float(dur[i]), "histogram", snap.version))
                add(MetricPoint(ts, service, M_SELF_DURATION, float(selfd[i]), "histogram", snap.version))
            if snap.dep_latency_ms > 0:
                depd = _lognormal_samples(rng, snap.dep_latency_ms, snap.sigma, DURATION_SAMPLES // 2)
                for i in range(len(depd)):
                    add(MetricPoint(ts_base + float(offsets[i]), service, M_DEP_DURATION,
                                    float(depd[i]), "histogram", snap.version))
            if snap.db_latency_ms > 0:
                dbd = _lognormal_samples(rng, snap.db_latency_ms, snap.sigma, DURATION_SAMPLES // 2)
                for i in range(len(dbd)):
                    add(MetricPoint(ts_base + float(offsets[i]), service, M_DB_DURATION,
                                    float(dbd[i]), "histogram", snap.version))

            requests = snap.rps * W
            errors = requests * snap.error_rate
            add(MetricPoint(ts_base, service, M_REQUESTS, requests, "counter", snap.version))
            add(MetricPoint(ts_base, service, M_ERRORS, errors, "counter", snap.version))
            add(MetricPoint(ts_base, service, M_TIMEOUTS, requests * snap.timeout_rate, "counter", snap.version))
            add(MetricPoint(ts_base, service, M_RETRIES, requests * snap.retry_rate, "counter", snap.version))
            add(MetricPoint(ts_base, service, M_RESTARTS, float(snap.restarts), "counter", snap.version))

            for k in range(3):
                ts = ts_base + (k + 0.5) * W / 3.0
                add(MetricPoint(ts, service, M_CPU,
                                float(np.clip(snap.cpu + rng.normal(0, 0.01), 0, 1)), "gauge", snap.version))
                add(MetricPoint(ts, service, M_MEMORY, snap.memory_mb, "gauge", snap.version))
                # Little's law: concurrency = arrival rate x residence time.
                add(MetricPoint(ts, service, M_CONCURRENT,
                                snap.rps * snap.total_latency_ms / 1000.0, "gauge", snap.version))
                if snap.pool_utilization > 0 or snap.pool_wait_ms > 0:
                    add(MetricPoint(ts, service, M_POOL_UTIL, snap.pool_utilization, "gauge", snap.version))
                    add(MetricPoint(ts, service, M_POOL_WAIT, snap.pool_wait_ms, "gauge", snap.version))

    def _emit_logs(
        self,
        batch: TelemetryBatch,
        snaps: Dict[str, ServiceSnapshot],
        effects: Dict[str, fault_lib.FaultEffect],
        t0: float,
    ) -> None:
        W = self.window_seconds
        rng = self.rng
        for service, snap in snaps.items():
            n_err = int(min(60, snap.rps * W * snap.error_rate * 0.25))
            event = _DEP_FAILURE_EVENT.get(snap.dominant_failing_dep or "", "request_failed")
            for _ in range(n_err):
                batch.logs.append(
                    LogRecord(
                        timestamp=t0 + float(rng.uniform(0, W)),
                        service=service,
                        severity=SEVERITY_ERROR,
                        event=event,
                        trace_id=new_id(),
                        attributes={"dependency": snap.dominant_failing_dep},
                    )
                )
            eff = effects.get(service)
            if eff:
                for severity, ev, count in eff.logs:
                    for _ in range(max(0, int(count))):
                        batch.logs.append(
                            LogRecord(
                                timestamp=t0 + float(rng.uniform(0, W)),
                                service=service,
                                severity=severity,
                                event=ev,
                                attributes=dict(eff.attributes),
                            )
                        )

    def _emit_traces(
        self, batch: TelemetryBatch, snaps: Dict[str, ServiceSnapshot], t0: float
    ) -> None:
        for _ in range(self.cfg.traces_per_window):
            checkout = self.rng.random() < 0.65
            ts = t0 + float(self.rng.uniform(0, self.window_seconds))
            batch.spans.extend(self._build_trace(snaps, ts, checkout))

    def _build_trace(
        self, snaps: Dict[str, ServiceSnapshot], ts: float, checkout: bool
    ) -> List[Span]:
        trace_id = new_id()
        spans: List[Span] = []

        def server_span(service: str, operation: str, parent: Optional[str], start: float):
            snap = snaps[service]
            self_ms = float(_lognormal_samples(self.rng, snap.self_latency_ms, snap.sigma, 1)[0])
            span = Span(
                trace_id=trace_id,
                span_id=new_id(),
                parent_span_id=parent,
                service=service,
                operation=operation,
                kind=SPAN_KIND_SERVER,
                start_time=start,
                duration_ms=self_ms,
                peer_service=None,
                attributes={"version": snap.version},
            )
            spans.append(span)
            return span, self_ms

        def call(caller_span: Span, callee: str, operation: str, start: float) -> float:
            e = topo.edge(caller_span.service, callee)
            snap = snaps[callee]
            child, child_self = server_span(callee, operation, None, start)
            # Recurse into the callee's own dependencies.
            child_total = child_self
            for grandchild in topo.callees(callee):
                ge = topo.edge(callee, grandchild)
                if self.rng.random() > min(1.0, ge.fanout):
                    continue
                child_total += call(child, grandchild, ge.operation, start + child_total)
            child.duration_ms = child_total

            timed_out = self.rng.random() < _p_exceed(
                snaps[callee].total_latency_ms, snap.sigma, e.timeout_ms
            ) ** (e.retries + 1)
            observed = min(child_total, e.timeout_ms) if not timed_out else e.timeout_ms * (e.retries + 1)
            client = Span(
                trace_id=trace_id,
                span_id=new_id(),
                parent_span_id=caller_span.span_id,
                service=caller_span.service,
                operation=f"{callee} {operation}",
                kind=SPAN_KIND_CLIENT,
                start_time=start,
                duration_ms=observed,
                status="TIMEOUT" if timed_out else "OK",
                peer_service=callee,
                attributes={"timeout_ms": e.timeout_ms, "retries": e.retries},
            )
            spans.append(client)
            child.parent_span_id = client.span_id
            if timed_out:
                child.status = "TIMEOUT"
            return observed

        operation = "POST /checkout" if checkout else "GET /products/{id}"
        root, root_self = server_span(topo.API_GATEWAY, operation, None, ts)
        total = root_self
        targets = (
            [(topo.ORDER_SERVICE, "POST /checkout")]
            if checkout
            else [(topo.INVENTORY_SERVICE, "GET /products/{id}")]
        )
        for callee, op in targets:
            total += call(root, callee, op, ts + total)
        root.duration_ms = total
        if any(s.status != "OK" for s in spans):
            root.status = "ERROR"
        return spans

    # -- labels ------------------------------------------------------------
    def affected_services(self, fault: ActiveFault, threshold: float = 1.6) -> List[str]:
        """Services whose modelled latency or error rate materially deviated
        from their pre-incident level during the fault."""
        pre_lo = max(0, fault.start_window - 16)
        pre_hi = fault.start_window
        hi = min(len(self.snapshots), fault.start_window + fault.duration_windows)
        if pre_hi <= pre_lo or hi <= fault.start_window:
            return [fault.origin]
        affected = []
        for s in topo.SERVICES:
            pre_lat = np.median([self.snapshots[i][s].total_latency_ms for i in range(pre_lo, pre_hi)])
            pre_err = np.median([self.snapshots[i][s].error_rate for i in range(pre_lo, pre_hi)])
            during = [self.snapshots[i][s] for i in range(fault.start_window, hi)]
            max_lat = max(d.total_latency_ms for d in during)
            max_err = max(d.error_rate for d in during)
            if max_lat > pre_lat * threshold or max_err > max(pre_err * 3.0, 0.02):
                affected.append(s)
        if fault.origin not in affected:
            affected.append(fault.origin)
        return affected


def _merge_effects(a: fault_lib.FaultEffect, b: fault_lib.FaultEffect) -> fault_lib.FaultEffect:
    return fault_lib.FaultEffect(
        self_latency_mult=a.self_latency_mult * b.self_latency_mult,
        self_latency_add_ms=a.self_latency_add_ms + b.self_latency_add_ms,
        self_latency_sigma_add=a.self_latency_sigma_add + b.self_latency_sigma_add,
        cpu_add=a.cpu_add + b.cpu_add,
        memory_add_mb=a.memory_add_mb + b.memory_add_mb,
        pool_utilization=b.pool_utilization if b.pool_utilization is not None else a.pool_utilization,
        pool_wait_ms=a.pool_wait_ms + b.pool_wait_ms,
        intrinsic_error_rate=1.0 - (1.0 - a.intrinsic_error_rate) * (1.0 - b.intrinsic_error_rate),
        capacity_mult=a.capacity_mult * b.capacity_mult,
        restart=a.restart or b.restart,
        logs=a.logs + b.logs,
        attributes={**a.attributes, **b.attributes},
        deploy_version=b.deploy_version or a.deploy_version,
    )
