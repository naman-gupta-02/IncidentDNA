"""Wire format for telemetry.

This is the contract between *producers* of telemetry (the real Dockerised
microservices via OpenTelemetry, or the simulator) and *consumers* (the
feature builder). Both paths emit exactly these four record types, which is
why the offline dataset and the live demo are scored by identical code.

Records are deliberately plain and slotted: a full offline dataset creates
~10^6 of them and pydantic models would dominate the runtime.
"""
from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

SPAN_KIND_SERVER = "server"
SPAN_KIND_CLIENT = "client"

SEVERITY_INFO = "INFO"
SEVERITY_WARN = "WARN"
SEVERITY_ERROR = "ERROR"


def new_id(n: int = 16) -> str:
    return uuid.uuid4().hex[:n]


@dataclass(slots=True)
class Span:
    """One unit of work. A server span is a request handled by a service;
    a client span is an outbound call that service made."""

    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    service: str
    operation: str
    kind: str
    #: Unix seconds.
    start_time: float
    duration_ms: float
    status: str = "OK"          # OK | ERROR | TIMEOUT
    #: The callee for client spans; None for server spans.
    peer_service: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)

    @property
    def end_time(self) -> float:
        return self.start_time + self.duration_ms / 1000.0

    @property
    def failed(self) -> bool:
        return self.status != "OK"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class MetricPoint:
    """A numeric measurement of a service at a point in time.

    `kind` distinguishes counters (summed over a window) from gauges
    (averaged / maxed over a window)."""

    timestamp: float
    service: str
    metric: str
    value: float
    kind: str = "gauge"          # gauge | counter
    version: str = "v1.0.0"
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class LogRecord:
    timestamp: float
    service: str
    severity: str
    event: str
    trace_id: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class DeploymentEvent:
    """A release. Temporal proximity to an anomaly is root-cause evidence."""

    timestamp: float
    service: str
    version: str
    previous_version: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class Trace:
    """A collected set of spans sharing a trace id."""

    trace_id: str
    spans: List[Span]

    @property
    def root(self) -> Span:
        for s in self.spans:
            if s.parent_span_id is None:
                return s
        return self.spans[0]

    @property
    def duration_ms(self) -> float:
        return self.root.duration_ms

    @property
    def start_time(self) -> float:
        return self.root.start_time

    @property
    def failed(self) -> bool:
        return any(s.failed for s in self.spans)

    def server_spans(self) -> List[Span]:
        return [s for s in self.spans if s.kind == SPAN_KIND_SERVER]

    def self_time_ms(self) -> Dict[str, float]:
        """Time attributable to each service: its server span duration minus
        the time it spent waiting on its own outbound calls.

        This is what makes a trace usable as root-cause evidence — a gateway
        span of 2.4s that spent 2.3s waiting on Postgres should credit
        Postgres, not the gateway."""
        by_parent: Dict[str, List[Span]] = {}
        for s in self.spans:
            if s.parent_span_id is not None:
                by_parent.setdefault(s.parent_span_id, []).append(s)
        out: Dict[str, float] = {}
        for s in self.server_spans():
            children = by_parent.get(s.span_id, [])
            child_time = sum(c.duration_ms for c in children if c.kind == SPAN_KIND_CLIENT)
            out[s.service] = out.get(s.service, 0.0) + max(0.0, s.duration_ms - child_time)
        return out

    def dominant_service(self) -> Optional[str]:
        """Service accounting for the largest share of wall-clock time."""
        st = self.self_time_ms()
        if not st:
            return None
        return max(st.items(), key=lambda kv: kv[1])[0]

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "duration_ms": self.duration_ms,
            "start_time": self.start_time,
            "failed": self.failed,
            "dominant_service": self.dominant_service(),
            "self_time_ms": self.self_time_ms(),
            "spans": [s.to_dict() for s in self.spans],
        }


@dataclass(slots=True)
class TelemetryBatch:
    """Everything emitted during one tick of the world."""

    spans: List[Span] = field(default_factory=list)
    metrics: List[MetricPoint] = field(default_factory=list)
    logs: List[LogRecord] = field(default_factory=list)
    deployments: List[DeploymentEvent] = field(default_factory=list)

    def extend(self, other: "TelemetryBatch") -> None:
        self.spans.extend(other.spans)
        self.metrics.extend(other.metrics)
        self.logs.extend(other.logs)
        self.deployments.extend(other.deployments)

    def traces(self) -> List[Trace]:
        by_trace: Dict[str, List[Span]] = {}
        for s in self.spans:
            by_trace.setdefault(s.trace_id, []).append(s)
        return [Trace(tid, spans) for tid, spans in by_trace.items()]

    def __len__(self) -> int:
        return len(self.spans) + len(self.metrics) + len(self.logs) + len(self.deployments)
