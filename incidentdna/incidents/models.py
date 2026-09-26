"""Incident data model."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from .. import topology as topo
from ..detect.base import ServiceAnomaly

STATUS_OPEN = "open"
STATUS_RESOLVED = "resolved"


def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def clock(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%H:%M:%S")


@dataclass
class ServiceIncidentState:
    """How one service behaved during one incident."""

    service: str
    onset_window: int
    onset_time: float
    peak_score: float = 0.0
    peak_window: int = 0
    last_anomalous_window: int = 0
    anomalous_windows: int = 0
    total_windows: int = 0
    peak_z: Dict[str, float] = field(default_factory=dict)
    scores: List[float] = field(default_factory=list)

    @property
    def duration_windows(self) -> int:
        return max(1, self.last_anomalous_window - self.onset_window + 1)

    @property
    def anomalous_fraction(self) -> float:
        return self.anomalous_windows / max(1, self.total_windows)

    def top_metrics(self, n: int = 3) -> List[tuple]:
        return sorted(self.peak_z.items(), key=lambda kv: -kv[1])[:n]

    def to_dict(self) -> dict:
        return {
            "service": self.service,
            "display_name": topo.DISPLAY_NAMES.get(self.service, self.service),
            "onset_window": self.onset_window,
            "onset_time": self.onset_time,
            "onset_clock": clock(self.onset_time),
            "peak_score": round(self.peak_score, 4),
            "anomalous_windows": self.anomalous_windows,
            "duration_windows": self.duration_windows,
            "top_metrics": [{"metric": m, "z": round(z, 2)} for m, z in self.top_metrics()],
        }


@dataclass
class Symptom:
    """The user-visible problem, stated in the metric's own units."""

    service: str
    metric: str
    baseline_value: float
    current_value: float
    #: Pre-rendered text from the evidence generator, when it found a metric
    #: that describes the symptom better than the default latency phrasing.
    text: Optional[str] = None

    def describe(self) -> str:
        if self.text:
            return self.text
        name = topo.DISPLAY_NAMES.get(self.service, self.service)
        pretty = self.metric.replace("_", " ")
        if self.metric.endswith("_rate"):
            return (
                f"{name} {pretty} rose from {self.baseline_value * 100:.1f}% "
                f"to {self.current_value * 100:.1f}%"
            )
        if "latency" in self.metric:
            return (
                f"{name} {pretty} rose from {_ms(self.baseline_value)} "
                f"to {_ms(self.current_value)}"
            )
        return (
            f"{name} {pretty} rose from {self.baseline_value:.2f} to {self.current_value:.2f}"
        )

    def to_dict(self) -> dict:
        return {
            "service": self.service,
            "metric": self.metric,
            "baseline_value": self.baseline_value,
            "current_value": self.current_value,
            "text": self.describe(),
        }


def _ms(v: float) -> str:
    return f"{v / 1000.0:.2f} s" if v >= 1000 else f"{v:.0f} ms"


@dataclass
class Candidate:
    """A service considered as the origin of an incident, with its evidence."""

    service: str
    features: Dict[str, float]
    score: float = 0.0
    rank: int = 0
    confidence: float = 0.0
    evidence: List[str] = field(default_factory=list)
    contradicting: List[str] = field(default_factory=list)
    contributions: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "service": self.service,
            "display_name": topo.DISPLAY_NAMES.get(self.service, self.service),
            "score": round(self.score, 4),
            "rank": self.rank,
            "confidence": round(self.confidence, 4),
            "features": {k: round(v, 4) for k, v in self.features.items()},
            "contributions": {k: round(v, 4) for k, v in self.contributions.items()},
            "evidence": self.evidence,
            "contradicting": self.contradicting,
        }


@dataclass
class Diagnosis:
    ranker: str
    candidates: List[Candidate]
    generated_at_window: int

    @property
    def root_cause(self) -> Optional[str]:
        return self.candidates[0].service if self.candidates else None

    @property
    def confidence(self) -> float:
        return self.candidates[0].confidence if self.candidates else 0.0

    def to_dict(self) -> dict:
        return {
            "ranker": self.ranker,
            "root_cause": self.root_cause,
            "root_cause_display": topo.DISPLAY_NAMES.get(self.root_cause, self.root_cause),
            "confidence": round(self.confidence, 4),
            "generated_at_window": self.generated_at_window,
            "candidates": [c.to_dict() for c in self.candidates],
        }


@dataclass
class TimelineEntry:
    window_index: int
    timestamp: float
    kind: str          # anomaly | deployment | detection | diagnosis | resolution
    service: Optional[str]
    text: str

    def to_dict(self) -> dict:
        return {
            "window_index": self.window_index,
            "timestamp": self.timestamp,
            "clock": clock(self.timestamp),
            "kind": self.kind,
            "service": self.service,
            "text": self.text,
        }


@dataclass
class Incident:
    incident_id: str
    start_window: int
    start_time: float
    detected_window: int
    detected_time: float
    status: str = STATUS_OPEN
    end_window: Optional[int] = None
    end_time: Optional[float] = None
    services: Dict[str, ServiceIncidentState] = field(default_factory=dict)
    symptom: Optional[Symptom] = None
    diagnosis: Optional[Diagnosis] = None
    timeline: List[TimelineEntry] = field(default_factory=list)
    severity: float = 0.0
    peak_score: float = 0.0
    #: Filled in only when a ground-truth label exists (simulated incidents).
    ground_truth: Optional[dict] = None

    @property
    def affected_services(self) -> List[str]:
        return sorted(self.services, key=lambda s: self.services[s].onset_window)

    @property
    def first_affected(self) -> Optional[str]:
        return self.affected_services[0] if self.services else None

    def severity_label(self) -> str:
        if self.severity >= 0.75:
            return "Critical"
        if self.severity >= 0.5:
            return "High"
        if self.severity >= 0.25:
            return "Medium"
        return "Low"

    def detection_delay_seconds(self, window_seconds: int) -> float:
        return (self.detected_window - self.start_window) * window_seconds

    def to_dict(self, window_seconds: int = 15) -> dict:
        return {
            "incident_id": self.incident_id,
            "status": self.status,
            "severity": round(self.severity, 3),
            "severity_label": self.severity_label(),
            "start_window": self.start_window,
            "start_time": self.start_time,
            "started_iso": iso(self.start_time),
            "started_clock": clock(self.start_time),
            "detected_window": self.detected_window,
            "detected_time": self.detected_time,
            "detected_clock": clock(self.detected_time),
            "detection_delay_seconds": self.detection_delay_seconds(window_seconds),
            "end_window": self.end_window,
            "end_time": self.end_time,
            "affected_services": self.affected_services,
            "services": {k: v.to_dict() for k, v in self.services.items()},
            "symptom": self.symptom.to_dict() if self.symptom else None,
            "diagnosis": self.diagnosis.to_dict() if self.diagnosis else None,
            "timeline": [t.to_dict() for t in self.timeline],
            "ground_truth": self.ground_truth,
        }

    def summary_text(self, window_seconds: int = 15) -> str:
        """The console-style diagnosis from section 1 of the guide."""
        lines = [
            f"Incident {self.incident_id}",
            f"Severity: {self.severity_label()}",
            f"Started: {clock(self.start_time)} UTC",
            "",
            f"Detected symptom: {self.symptom.describe() if self.symptom else 'n/a'}",
        ]
        if self.diagnosis and self.diagnosis.candidates:
            top = self.diagnosis.candidates[0]
            lines.append(
                f"Most likely root cause: {topo.DISPLAY_NAMES.get(top.service, top.service)}"
            )
            lines.append(f"Confidence: {top.confidence * 100:.0f}%")
            lines.append("")
            lines.append("Evidence:")
            for i, ev in enumerate(top.evidence, 1):
                lines.append(f"{i}. {ev}")
            if top.contradicting:
                lines.append("")
                lines.append("Contradicting evidence:")
                for i, ev in enumerate(top.contradicting, 1):
                    lines.append(f"{i}. {ev}")
        return "\n".join(lines)
