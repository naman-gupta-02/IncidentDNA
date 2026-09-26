"""Stage-one evaluation: incident-level, not row-level.

Row accuracy is the wrong unit. An on-call engineer cares how many real
incidents were caught, how fast, and how much noise arrived in between — so
alerts are matched to injected incidents and scored as whole events.

Matching rule: an alert matches an injected incident when the alert fires
between the incident's start and its end plus a grace period. An incident can
be matched at most once; extra concurrent alerts in the same run count as
duplicates, not as new false alerts, because on-call would see one page.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..config import WINDOW_SECONDS


@dataclass
class AlertRecord:
    run_id: str
    incident_id: str
    detected_window: int
    start_window: int
    end_window: Optional[int]
    severity: float


@dataclass
class TruthRecord:
    run_id: str
    fault_type: str
    root_cause_service: str
    start_window: int
    end_window: int


@dataclass
class DetectionResult:
    matched: List[Dict] = field(default_factory=list)
    missed: List[Dict] = field(default_factory=list)
    false_alerts: List[Dict] = field(default_factory=list)
    duplicate_alerts: int = 0
    normal_window_count: int = 0
    window_seconds: int = WINDOW_SECONDS

    # -- metrics -----------------------------------------------------------
    @property
    def true_positives(self) -> int:
        return len(self.matched)

    @property
    def precision(self) -> float:
        total = self.true_positives + len(self.false_alerts)
        return self.true_positives / total if total else 0.0

    @property
    def recall(self) -> float:
        total = self.true_positives + len(self.missed)
        return self.true_positives / total if total else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def detection_delays(self) -> List[float]:
        return [m["delay_seconds"] for m in self.matched]

    @property
    def mean_detection_delay(self) -> float:
        d = self.detection_delays
        return float(np.mean(d)) if d else float("nan")

    @property
    def median_detection_delay(self) -> float:
        d = self.detection_delays
        return float(np.median(d)) if d else float("nan")

    @property
    def p95_detection_delay(self) -> float:
        d = self.detection_delays
        return float(np.percentile(d, 95)) if d else float("nan")

    @property
    def normal_hours(self) -> float:
        return self.normal_window_count * self.window_seconds / 3600.0

    @property
    def false_alerts_per_hour(self) -> float:
        return len(self.false_alerts) / self.normal_hours if self.normal_hours > 0 else 0.0

    def by_fault_type(self) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for group, key in ((self.matched, "detected"), (self.missed, "missed")):
            for rec in group:
                ft = rec["fault_type"]
                out.setdefault(ft, {"detected": 0, "missed": 0, "delays": []})
                out[ft][key] += 1
                if key == "detected":
                    out[ft]["delays"].append(rec["delay_seconds"])
        summary = {}
        for ft, v in out.items():
            total = v["detected"] + v["missed"]
            summary[ft] = {
                "recall": v["detected"] / total if total else 0.0,
                "n": total,
                "median_delay_seconds": float(np.median(v["delays"])) if v["delays"] else float("nan"),
            }
        return summary

    def to_dict(self) -> dict:
        return {
            "incident_precision": round(self.precision, 4),
            "incident_recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "true_positives": self.true_positives,
            "false_alerts": len(self.false_alerts),
            "missed": len(self.missed),
            "duplicate_alerts": self.duplicate_alerts,
            "mean_detection_delay_seconds": _round(self.mean_detection_delay),
            "median_detection_delay_seconds": _round(self.median_detection_delay),
            "p95_detection_delay_seconds": _round(self.p95_detection_delay),
            "false_alerts_per_hour": round(self.false_alerts_per_hour, 4),
            "normal_hours_observed": round(self.normal_hours, 3),
            "by_fault_type": self.by_fault_type(),
        }


def _round(v: float, n: int = 2) -> Optional[float]:
    return None if v != v else round(v, n)   # NaN check


def evaluate_detection(
    alerts_by_run: Dict[str, List[AlertRecord]],
    truth_by_run: Dict[str, Optional[TruthRecord]],
    normal_windows_by_run: Dict[str, int],
    grace_windows: int = 8,
    window_seconds: int = WINDOW_SECONDS,
) -> DetectionResult:
    result = DetectionResult(window_seconds=window_seconds)
    for run_id, truth in truth_by_run.items():
        alerts = sorted(alerts_by_run.get(run_id, []), key=lambda a: a.detected_window)
        if truth is None:
            for a in alerts:
                result.false_alerts.append(
                    {"run_id": run_id, "incident_id": a.incident_id,
                     "detected_window": a.detected_window, "reason": "no fault injected"}
                )
            result.normal_window_count += normal_windows_by_run.get(run_id, 0)
            continue

        window_lo = truth.start_window
        window_hi = truth.end_window + grace_windows
        hit = None
        for a in alerts:
            if window_lo <= a.detected_window <= window_hi:
                if hit is None:
                    hit = a
                else:
                    result.duplicate_alerts += 1
            else:
                result.false_alerts.append(
                    {"run_id": run_id, "incident_id": a.incident_id,
                     "detected_window": a.detected_window,
                     "reason": "outside injected incident"}
                )
        # Windows in a fault run that lie outside the incident still count as
        # normal observation time for the noise rate.
        result.normal_window_count += max(
            0, normal_windows_by_run.get(run_id, 0) - (truth.end_window - truth.start_window)
        )
        if hit is not None:
            result.matched.append(
                {
                    "run_id": run_id,
                    "incident_id": hit.incident_id,
                    "fault_type": truth.fault_type,
                    "root_cause_service": truth.root_cause_service,
                    "delay_seconds": (hit.detected_window - truth.start_window) * window_seconds,
                    "detected_window": hit.detected_window,
                    "truth_start_window": truth.start_window,
                }
            )
        else:
            result.missed.append(
                {
                    "run_id": run_id,
                    "fault_type": truth.fault_type,
                    "root_cause_service": truth.root_cause_service,
                    "truth_start_window": truth.start_window,
                }
            )
    return result
