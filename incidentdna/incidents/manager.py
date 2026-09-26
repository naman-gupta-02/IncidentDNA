"""Incident lifecycle.

Per-window anomaly scores are noisy; incidents are not. This module debounces
scores into confirmed service anomalies, groups concurrent anomalies into one
incident, keeps the timeline, and re-runs the root-cause ranker as new
evidence arrives.

Debouncing rules (all in `DetectionConfig`):
  * a service is confirmed after `consecutive_windows` windows at or above
    `service_score_threshold`, and its onset is backdated to the first of them;
  * an incident opens on the first confirmation and adopts every service
    confirmed while it is open;
  * it resolves once everything has been below `clear_threshold` for
    `close_after_quiet_windows`.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional, Sequence

from .. import topology as topo
from ..config import SETTINGS
from ..detect.base import ServiceAnomaly
from ..pipeline.store import FeatureStore
from ..rca.candidates import IncidentContext
from ..rca.evidence import build_symptom_text
from ..rca.ranker import RootCauseRanker, default_ranker
from ..telemetry.schema import DeploymentEvent, Trace
from .models import (
    Incident,
    STATUS_OPEN,
    STATUS_RESOLVED,
    ServiceIncidentState,
    Symptom,
    TimelineEntry,
    clock,
)

_D = SETTINGS.detection


class IncidentManager:
    def __init__(
        self,
        store: FeatureStore,
        ranker: Optional[RootCauseRanker] = None,
        window_seconds: int = 15,
        rediagnose_every: int = 2,
    ) -> None:
        self.store = store
        self.ranker = ranker or default_ranker()
        self.window_seconds = window_seconds
        self.rediagnose_every = rediagnose_every

        self.incidents: List[Incident] = []
        self.open_incident: Optional[Incident] = None
        self.deployments: List[DeploymentEvent] = []
        self._traces: List[Trace] = []
        self._streaks: Dict[str, int] = defaultdict(int)
        self._pending: Dict[str, List[ServiceAnomaly]] = defaultdict(list)
        self._quiet_windows = 0
        self._counter = 0
        self._last_diagnosed_window = -999
        #: Every window's scores, kept for the dashboard's charts.
        self.recent_scores: Deque[Dict] = deque(maxlen=2000)
        #: Full scoring history per service, including sub-threshold windows.
        #: Needed to build candidate rows for services that never tripped the
        #: alert but sit underneath one that did.
        self._history: Dict[str, Deque[ServiceAnomaly]] = defaultdict(
            lambda: deque(maxlen=600)
        )

    # -- inputs ------------------------------------------------------------
    def record_deployments(self, deployments: Sequence[DeploymentEvent]) -> None:
        self.deployments.extend(deployments)
        for d in deployments:
            if self.open_incident is not None:
                self.open_incident.timeline.append(
                    TimelineEntry(
                        window_index=int(d.timestamp // self.window_seconds),
                        timestamp=d.timestamp,
                        kind="deployment",
                        service=d.service,
                        text=f"{topo.DISPLAY_NAMES.get(d.service, d.service)} deployed "
                        f"{d.previous_version} to {d.version}",
                    )
                )

    def record_traces(self, traces: Sequence[Trace]) -> None:
        self._traces.extend(traces)
        if len(self._traces) > 4000:
            self._traces = self._traces[-4000:]

    # -- the tick ----------------------------------------------------------
    def update(self, window_index: int, anomalies: Sequence[ServiceAnomaly]) -> Optional[Incident]:
        """Feed one window's scores. Returns the open incident, if any."""
        self.recent_scores.append(
            {
                "window_index": window_index,
                "scores": {a.service: round(a.score, 4) for a in anomalies},
            }
        )
        for a in anomalies:
            self._history[a.service].append(a)
        confirmed = self._confirm(anomalies)
        any_elevated = any(a.score >= _D.clear_threshold for a in anomalies)

        if confirmed and self.open_incident is None:
            self._open_incident(window_index, confirmed)
        elif confirmed and self.open_incident is not None:
            self._adopt(self.open_incident, confirmed)

        if self.open_incident is not None:
            self._track(self.open_incident, anomalies)
            if any_elevated:
                self._quiet_windows = 0
            else:
                self._quiet_windows += 1
                if self._quiet_windows >= _D.close_after_quiet_windows:
                    self._close_incident(window_index)
                    return None
            if window_index - self._last_diagnosed_window >= self.rediagnose_every:
                self.diagnose(self.open_incident)
                self._last_diagnosed_window = window_index
        return self.open_incident

    def _confirm(self, anomalies: Sequence[ServiceAnomaly]) -> List[ServiceAnomaly]:
        """Apply the consecutive-window debounce; return newly confirmed ones
        carrying the anomaly record from their *first* elevated window."""
        confirmed: List[ServiceAnomaly] = []
        seen = set()
        for a in anomalies:
            seen.add(a.service)
            if a.is_anomalous:
                self._streaks[a.service] += 1
                self._pending[a.service].append(a)
                if self._streaks[a.service] == _D.consecutive_windows:
                    confirmed.append(self._pending[a.service][0])
            else:
                self._streaks[a.service] = 0
                self._pending[a.service] = []
        for service in list(self._streaks):
            if service not in seen:
                self._streaks[service] = 0
                self._pending[service] = []
        return confirmed

    def _open_incident(self, window_index: int, confirmed: List[ServiceAnomaly]) -> None:
        self._counter += 1
        first = min(confirmed, key=lambda a: a.window_index)
        incident = Incident(
            incident_id=f"inc_{self._counter:05d}",
            start_window=first.window_index,
            start_time=first.window_start,
            detected_window=window_index,
            detected_time=window_index * self.window_seconds,
        )
        incident.timeline.append(
            TimelineEntry(
                window_index=window_index,
                timestamp=incident.detected_time,
                kind="detection",
                service=None,
                text=f"Incident detected ({len(confirmed)} service(s) abnormal)",
            )
        )
        self.open_incident = incident
        self.incidents.append(incident)
        self._quiet_windows = 0
        self._last_diagnosed_window = -999
        self._adopt(incident, confirmed)

    def _adopt(self, incident: Incident, confirmed: List[ServiceAnomaly]) -> None:
        for a in confirmed:
            if a.service in incident.services:
                continue
            incident.services[a.service] = ServiceIncidentState(
                service=a.service,
                onset_window=a.window_index,
                onset_time=a.window_start,
                peak_score=a.score,
                peak_window=a.window_index,
                last_anomalous_window=a.window_index,
                peak_z=dict(a.z_scores),
            )
            top = a.top_metrics(1)
            detail = f" ({top[0][0]} z={top[0][1]:.1f})" if top else ""
            incident.timeline.append(
                TimelineEntry(
                    window_index=a.window_index,
                    timestamp=a.window_start,
                    kind="anomaly",
                    service=a.service,
                    text=f"{topo.DISPLAY_NAMES.get(a.service, a.service)} became abnormal{detail}",
                )
            )
            if a.window_index < incident.start_window:
                incident.start_window = a.window_index
                incident.start_time = a.window_start
        incident.timeline.sort(key=lambda t: (t.timestamp, t.kind))

    def _track(self, incident: Incident, anomalies: Sequence[ServiceAnomaly]) -> None:
        for a in anomalies:
            st = incident.services.get(a.service)
            if st is None:
                continue
            st.total_windows += 1
            st.scores.append(a.score)
            if a.score > st.peak_score:
                st.peak_score = a.score
                st.peak_window = a.window_index
                st.peak_z = dict(a.z_scores)
            if a.is_anomalous:
                st.anomalous_windows += 1
                st.last_anomalous_window = a.window_index
            incident.peak_score = max(incident.peak_score, a.score)
        incident.severity = self._severity(incident)

    def _severity(self, incident: Incident) -> float:
        """Weighted by user impact: errors at the entrypoint dominate."""
        peak = incident.peak_score
        breadth = len(incident.services) / max(1, len(topo.SERVICES))
        user_impact = 0.0
        for ep in topo.ENTRYPOINTS:
            if ep in incident.services:
                hi = incident.end_window or self.store.latest_window_index
                vals = [
                    w.get("error_rate")
                    for w in self.store.history(ep)
                    if incident.start_window <= w.window_index <= hi
                ]
                if vals:
                    user_impact = max(user_impact, min(1.0, max(vals) / 0.2))
        return float(min(1.0, 0.45 * peak + 0.2 * breadth + 0.35 * user_impact))

    def _close_incident(self, window_index: int) -> None:
        incident = self.open_incident
        if incident is None:
            return
        incident.status = STATUS_RESOLVED
        incident.end_window = window_index
        incident.end_time = window_index * self.window_seconds
        incident.timeline.append(
            TimelineEntry(
                window_index=window_index,
                timestamp=incident.end_time,
                kind="resolution",
                service=None,
                text="All services returned to normal",
            )
        )
        self.diagnose(incident)
        self.open_incident = None
        self._quiet_windows = 0

    # -- diagnosis ---------------------------------------------------------
    def candidate_pool(self, incident: Incident) -> Dict[str, ServiceIncidentState]:
        """Confirmed services, plus every direct dependency of one.

        A cause sits *underneath* its symptoms, and it does not have to shout
        as loudly as they do: a database that slowed from 9 ms to 120 ms may
        stay under the alert threshold while the checkout it blocks times out
        spectacularly. Including quiet dependencies as candidates is what
        stops those incidents from being unrankable — the cost is a wider
        candidate set, which the evaluation reports as `mean_candidates`.
        """
        pool: Dict[str, ServiceIncidentState] = dict(incident.services)
        lo = incident.start_window
        hi = incident.end_window or self.store.latest_window_index
        for service in list(incident.services):
            for dep in topo.callees(service):
                if dep in pool:
                    continue
                state = self._quiet_state(dep, lo, hi)
                if state is not None:
                    pool[dep] = state
        return pool

    def _quiet_state(
        self, service: str, lo: int, hi: int
    ) -> Optional[ServiceIncidentState]:
        """Build a candidate state for a service that never tripped the alert."""
        history = [a for a in self._history.get(service, ()) if lo <= a.window_index <= hi]
        if not history:
            return None
        peak = max(history, key=lambda a: a.score)
        # With no threshold crossing, "onset" is the first window where the
        # service reached half of its own peak deviation.
        onset = next(
            (a for a in history if a.score >= _D.clear_threshold),
            next((a for a in history if a.score >= 0.5 * peak.score), peak),
        )
        return ServiceIncidentState(
            service=service,
            onset_window=onset.window_index,
            onset_time=onset.window_start,
            peak_score=peak.score,
            peak_window=peak.window_index,
            last_anomalous_window=peak.window_index,
            anomalous_windows=sum(1 for a in history if a.is_anomalous),
            total_windows=len(history),
            peak_z=dict(peak.z_scores),
            scores=[a.score for a in history],
        )

    def context(self, incident: Incident) -> IncidentContext:
        lo = incident.start_time - 60
        hi = (incident.end_time or (self.store.latest_window_index * self.window_seconds)) + 60
        return IncidentContext(
            incident=incident,
            store=self.store,
            deployments=self.deployments,
            traces=[t for t in self._traces if lo <= t.start_time <= hi],
            window_seconds=self.window_seconds,
            states=self.candidate_pool(incident),
            confirmed=frozenset(incident.services),
        )

    def diagnose(
        self, incident: Incident, ranker: Optional[RootCauseRanker] = None
    ) -> Optional[Incident]:
        ctx = self.context(incident)
        diagnosis = (ranker or self.ranker).diagnose(ctx)
        if diagnosis is None:
            return incident
        first_diagnosis = incident.diagnosis is None
        previous = incident.diagnosis.root_cause if incident.diagnosis else None
        incident.diagnosis = diagnosis
        symptom_text = build_symptom_text(ctx)
        if symptom_text:
            symptom_service = next(
                (s for s in topo.ENTRYPOINTS if s in incident.services),
                max(incident.services, key=lambda s: incident.services[s].peak_score),
            )
            metric = "latency_p95"
            incident.symptom = Symptom(
                service=symptom_service,
                metric=metric,
                baseline_value=ctx.baseline(symptom_service, metric),
                current_value=ctx.peak_value(symptom_service, metric),
                text=symptom_text,
            )
        if first_diagnosis or previous != diagnosis.root_cause:
            incident.timeline.append(
                TimelineEntry(
                    window_index=diagnosis.generated_at_window,
                    timestamp=diagnosis.generated_at_window * self.window_seconds,
                    kind="diagnosis",
                    service=diagnosis.root_cause,
                    text=f"{topo.DISPLAY_NAMES.get(diagnosis.root_cause, diagnosis.root_cause)} "
                    f"ranked as root cause ({diagnosis.confidence * 100:.0f}% confidence)",
                )
            )
            incident.timeline.sort(key=lambda t: (t.timestamp, t.kind))
        return incident

    def rediagnose_all(self, ranker: RootCauseRanker) -> None:
        for incident in self.incidents:
            self.diagnose(incident, ranker=ranker)

    # -- accessors ---------------------------------------------------------
    def by_id(self, incident_id: str) -> Optional[Incident]:
        return next((i for i in self.incidents if i.incident_id == incident_id), None)

    def as_list(self) -> List[dict]:
        return [i.to_dict(self.window_seconds) for i in reversed(self.incidents)]
