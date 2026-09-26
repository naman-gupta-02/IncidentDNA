"""Instrumentation shared by the real microservices.

Produces exactly the records in `incidentdna.telemetry.schema`, so the feature
pipeline cannot tell whether it is reading the simulator or the Dockerised
services. Trace context is propagated with a W3C `traceparent` header.

Counters and gauges are aggregated in-process and flushed once a second;
latency is exported as sampled duration exemplars, which is the same split the
simulator uses and the same split a real Prometheus + tracing stack uses.

If no Kafka broker is reachable the emitter degrades to a no-op with a single
warning, so a service still starts and serves traffic.
"""
from __future__ import annotations

import contextlib
import contextvars
import logging
import os
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional

from incidentdna.pipeline.features import (
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
from incidentdna.telemetry.schema import (
    LogRecord,
    MetricPoint,
    SEVERITY_ERROR,
    SEVERITY_WARN,
    SPAN_KIND_CLIENT,
    SPAN_KIND_SERVER,
    Span,
    TelemetryBatch,
    new_id,
)

log = logging.getLogger("incidentdna.telemetry")

#: Fraction of requests that carry a sampled trace. Metrics are exact; traces
#: are sampled, exactly as in a production tracing deployment.
TRACE_SAMPLE_RATE = float(os.environ.get("TRACE_SAMPLE_RATE", "0.25"))
FLUSH_INTERVAL_SECONDS = 1.0

_current_span: contextvars.ContextVar[Optional[Span]] = contextvars.ContextVar(
    "incidentdna_current_span", default=None
)
_current_trace: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "incidentdna_current_trace", default=None
)
_sampled: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "incidentdna_sampled", default=False
)


def parse_traceparent(header: Optional[str]) -> tuple:
    """(trace_id, parent_span_id, sampled) from a W3C traceparent header."""
    if not header:
        return None, None, None
    parts = header.split("-")
    if len(parts) != 4:
        return None, None, None
    _, trace_id, span_id, flags = parts
    return trace_id, span_id, flags.endswith("1")


def format_traceparent(trace_id: str, span_id: str, sampled: bool) -> str:
    return f"00-{trace_id}-{span_id}-{'01' if sampled else '00'}"


@dataclass
class _Counters:
    requests: float = 0.0
    errors: float = 0.0
    timeouts: float = 0.0
    retries: float = 0.0
    restarts: float = 0.0


class Telemetry:
    """Per-service emitter. One instance per process."""

    def __init__(self, service: str, version: str = "v1.0.0", bus=None) -> None:
        self.service = service
        self.version = version
        self.bus = bus if bus is not None else _make_bus()
        self._lock = threading.Lock()
        self._pending = TelemetryBatch()
        self._counters = _Counters()
        self.in_flight = 0
        #: Set by the fault controller so pool metrics reflect injected faults.
        self.pool_utilization = 0.0
        self.pool_wait_ms = 0.0
        self.extra_memory_mb = 0.0
        self._stop = threading.Event()
        self._flusher = threading.Thread(target=self._flush_loop, daemon=True)
        self._flusher.start()

    # -- emission ----------------------------------------------------------
    def _add_metric(self, name: str, value: float, kind: str = "gauge") -> None:
        with self._lock:
            self._pending.metrics.append(
                MetricPoint(time.time(), self.service, name, float(value), kind, self.version)
            )

    def emit_span(self, span: Span) -> None:
        with self._lock:
            self._pending.spans.append(span)

    def emit_log(self, severity: str, event: str, **attributes) -> None:
        with self._lock:
            self._pending.logs.append(
                LogRecord(
                    timestamp=time.time(),
                    service=self.service,
                    severity=severity,
                    event=event,
                    trace_id=_current_trace.get(),
                    attributes=attributes,
                )
            )

    def emit_deployment(self, version: str) -> None:
        from incidentdna.telemetry.schema import DeploymentEvent

        previous, self.version = self.version, version
        with self._lock:
            self._pending.deployments.append(
                DeploymentEvent(time.time(), self.service, version, previous)
            )

    def record_restart(self) -> None:
        with self._lock:
            self._counters.restarts += 1

    # -- request lifecycle -------------------------------------------------
    @contextlib.contextmanager
    def server_span(self, operation: str, traceparent: Optional[str] = None) -> Iterator[Span]:
        trace_id, parent_span_id, sampled = parse_traceparent(traceparent)
        if trace_id is None:
            trace_id = new_id(32)
            sampled = random.random() < TRACE_SAMPLE_RATE
        span = Span(
            trace_id=trace_id,
            span_id=new_id(16),
            parent_span_id=parent_span_id,
            service=self.service,
            operation=operation,
            kind=SPAN_KIND_SERVER,
            start_time=time.time(),
            duration_ms=0.0,
            attributes={"version": self.version},
        )
        tokens = (
            _current_span.set(span),
            _current_trace.set(trace_id),
            _sampled.set(bool(sampled)),
        )
        with self._lock:
            self.in_flight += 1
            self._counters.requests += 1
        child_ms = 0.0
        try:
            yield span
        except Exception:
            span.status = "ERROR"
            with self._lock:
                self._counters.errors += 1
            raise
        finally:
            span.duration_ms = (time.time() - span.start_time) * 1000.0
            child_ms = float(span.attributes.pop("_child_ms", 0.0))
            self_ms = max(0.0, span.duration_ms - child_ms)
            self._add_metric(M_DURATION, span.duration_ms, "histogram")
            self._add_metric(M_SELF_DURATION, self_ms, "histogram")
            if child_ms > 0:
                self._add_metric(M_DEP_DURATION, child_ms, "histogram")
            if span.status != "OK":
                with self._lock:
                    self._counters.errors += 1
            if _sampled.get():
                self.emit_span(span)
            with self._lock:
                self.in_flight -= 1
            for var, token in zip((_current_span, _current_trace, _sampled), tokens):
                var.reset(token)

    @contextlib.contextmanager
    def client_span(self, peer: str, operation: str, timeout_ms: float, retries: int) -> Iterator[Span]:
        parent = _current_span.get()
        span = Span(
            trace_id=_current_trace.get() or new_id(32),
            span_id=new_id(16),
            parent_span_id=parent.span_id if parent else None,
            service=self.service,
            operation=f"{peer} {operation}",
            kind=SPAN_KIND_CLIENT,
            start_time=time.time(),
            duration_ms=0.0,
            peer_service=peer,
            attributes={"timeout_ms": timeout_ms, "retries": retries},
        )
        try:
            yield span
        finally:
            span.duration_ms = (time.time() - span.start_time) * 1000.0
            if parent is not None:
                parent.attributes["_child_ms"] = (
                    float(parent.attributes.get("_child_ms", 0.0)) + span.duration_ms
                )
            if span.status == "TIMEOUT":
                with self._lock:
                    self._counters.timeouts += 1
            if _sampled.get():
                self.emit_span(span)

    def record_retry(self, n: int = 1) -> None:
        with self._lock:
            self._counters.retries += n

    def record_db_query(self, duration_ms: float) -> None:
        self._add_metric(M_DB_DURATION, duration_ms, "histogram")

    def traceparent(self) -> Optional[str]:
        span = _current_span.get()
        if span is None:
            return None
        return format_traceparent(span.trace_id, span.span_id, _sampled.get())

    # -- background flush --------------------------------------------------
    def _resource_sample(self) -> None:
        cpu = memory = None
        try:
            import psutil  # type: ignore

            proc = psutil.Process()
            cpu = proc.cpu_percent(interval=None) / 100.0
            memory = proc.memory_info().rss / (1024 * 1024)
        except Exception:
            try:
                import resource

                memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
            except Exception:
                memory = 0.0
            cpu = min(1.0, os.getloadavg()[0] / max(1, os.cpu_count() or 1))
        self._add_metric(M_CPU, max(0.0, min(1.0, cpu or 0.0)))
        self._add_metric(M_MEMORY, (memory or 0.0) + self.extra_memory_mb)
        self._add_metric(M_CONCURRENT, float(self.in_flight))
        if self.pool_utilization or self.pool_wait_ms:
            self._add_metric(M_POOL_UTIL, self.pool_utilization)
            self._add_metric(M_POOL_WAIT, self.pool_wait_ms)

    def flush(self) -> None:
        self._resource_sample()
        with self._lock:
            c, self._counters = self._counters, _Counters()
            now = time.time()
            for name, value in (
                (M_REQUESTS, c.requests),
                (M_ERRORS, c.errors),
                (M_TIMEOUTS, c.timeouts),
                (M_RETRIES, c.retries),
                (M_RESTARTS, c.restarts),
            ):
                self._pending.metrics.append(
                    MetricPoint(now, self.service, name, value, "counter", self.version)
                )
            batch, self._pending = self._pending, TelemetryBatch()
        if len(batch):
            try:
                self.bus.publish(batch)
            except Exception:  # pragma: no cover - broker hiccup
                log.warning("telemetry publish failed", exc_info=False)

    def _flush_loop(self) -> None:
        while not self._stop.wait(FLUSH_INTERVAL_SECONDS):
            try:
                self.flush()
            except Exception:  # pragma: no cover
                log.exception("flush loop error")

    def close(self) -> None:
        self._stop.set()
        self.flush()
        with contextlib.suppress(Exception):
            self.bus.close()


class _NullBus:
    def publish(self, batch) -> None:
        return None

    def drain(self):
        return TelemetryBatch()

    def close(self) -> None:
        return None


def _make_bus():
    """Kafka when a broker is configured and reachable, a JSONL capture file
    when `TELEMETRY_FILE` is set, otherwise a no-op.

    The file sink is what lets the services be run and verified with no
    infrastructure at all: start them, drive traffic, then replay the capture
    through the same pipeline the live system uses."""
    capture = os.environ.get("TELEMETRY_FILE")
    if capture:
        from incidentdna.telemetry.bus import FileBus

        log.info("telemetry capture -> %s", capture)
        return FileBus(capture, mode="a")

    servers = os.environ.get("KAFKA_BOOTSTRAP_SERVERS")
    if not servers:
        log.warning("KAFKA_BOOTSTRAP_SERVERS unset; telemetry will be discarded")
        return _NullBus()
    try:
        from incidentdna.telemetry.bus import KafkaBus

        return KafkaBus(bootstrap_servers=servers, consume=False, produce=True)
    except Exception as exc:
        log.warning("Kafka unavailable (%s); telemetry will be discarded", exc)
        return _NullBus()
