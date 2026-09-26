"""Stage two, part one: turn an incident into one feature row per candidate.

Every service that went abnormal during an incident becomes a candidate. The
features below are the ones that actually separate a cause from a symptom:

* **onset order** - a cause is abnormal before the things it breaks;
* **self vs dependency latency** - a cause's *own* work got slower, a symptom
  is just waiting on somebody else;
* **propagation consistency** - the services that went bad after it should be
  the ones that call it, directly or transitively;
* **trace attribution** - in slow traces, the cause holds the wall clock;
* **change and log proximity** - a release or an error burst at the onset.

Each row is scored either by the transparent weighted baseline
(`rca.baseline`) or by a learned ranker (`rca.learned`); both read the same
columns, so their outputs are directly comparable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .. import topology as topo
from ..config import SETTINGS
from ..incidents.models import Incident, ServiceIncidentState
from ..pipeline.store import FeatureStore
from ..telemetry.schema import DeploymentEvent, Trace

EPS = 1e-9

#: The candidate feature vector, in a fixed order.
CANDIDATE_FEATURES: List[str] = [
    "anomaly_strength",
    "anomaly_duration",
    "early_onset",
    "onset_lead_seconds_norm",
    "propagation_consistency",
    "downstream_affected_ratio",
    "dependency_broke_first",
    "self_latency_share",
    "error_originated",
    "trace_attribution",
    "deployment_evidence",
    "log_evidence",
    "graph_depth_norm",
    "is_leaf",
]


@dataclass
class IncidentContext:
    """Everything the ranker is allowed to look at for one incident.

    `states` is the *candidate pool* and is usually wider than the set of
    services that actually tripped the alert threshold: a dependency of an
    abnormal service is a candidate even when it stayed under the threshold
    itself, because a cause is often subtler than the symptom it produces.
    `confirmed` is the narrower set that did trip, and is what the
    propagation-consistency reasoning treats as observed symptoms.
    """

    incident: Incident
    store: FeatureStore
    deployments: Sequence[DeploymentEvent] = ()
    traces: Sequence[Trace] = ()
    window_seconds: int = 15
    states: Dict[str, ServiceIncidentState] = field(default_factory=dict)
    confirmed: frozenset = frozenset()

    def __post_init__(self) -> None:
        if not self.states:
            self.states = dict(self.incident.services)
        if not self.confirmed:
            self.confirmed = frozenset(self.incident.services)

    def baseline(self, service: str, metric: str) -> float:
        b = self.store.robust_baseline(
            service, metric, self.incident.start_window, lookback=24
        )
        if b is None:
            b = self.store.robust_baseline(service, metric, self.incident.start_window, lookback=64)
        return b[0] if b else 0.0

    def peak_value(self, service: str, metric: str) -> float:
        lo = self.incident.start_window
        hi = self.incident.end_window or self.store.latest_window_index
        vals = [
            w.get(metric)
            for w in self.store.history(service)
            if lo <= w.window_index <= hi
        ]
        return max(vals) if vals else 0.0

    def mean_value(self, service: str, metric: str) -> float:
        lo = self.incident.start_window
        hi = self.incident.end_window or self.store.latest_window_index
        vals = [
            w.get(metric)
            for w in self.store.history(service)
            if lo <= w.window_index <= hi
        ]
        return float(np.mean(vals)) if vals else 0.0

    def delta(self, service: str, metric: str) -> float:
        return max(0.0, self.peak_value(service, metric) - self.baseline(service, metric))

    def last_deployment_before(self, service: str, ts: float) -> Optional[DeploymentEvent]:
        best = None
        for d in self.deployments:
            if d.service == service and d.timestamp <= ts:
                if best is None or d.timestamp > best.timestamp:
                    best = d
        return best


def _safe_ratio(num: float, den: float) -> float:
    return float(num / den) if den > EPS else 0.0


def _propagation_consistency(
    service: str, states: Dict[str, ServiceIncidentState], confirmed: frozenset
) -> tuple:
    """(score in [0,1], supporting, contradicting, deps_before).

    A candidate is supported when the services that *call* it went abnormal
    after it, and undermined when a caller went abnormal first or when one of
    its own dependencies broke before it did. Only services that actually
    tripped the threshold count as observed symptoms.
    """
    onset = states[service].onset_window
    callers = topo.transitive_callers(service)
    callees = topo.transitive_callees(service)
    supporting = contradicting = deps_before = 0
    others = [s for s in confirmed if s != service and s in states]
    for other in others:
        o_onset = states[other].onset_window
        if other in callers:
            if o_onset >= onset:
                supporting += 1
            else:
                contradicting += 1
        elif other in callees:
            if o_onset < onset:
                deps_before += 1
    raw = (supporting - contradicting - 1.5 * deps_before) / max(1, len(others))
    return float(np.clip((raw + 1.0) / 2.0, 0.0, 1.0)), supporting, contradicting, deps_before


def _trace_attribution(ctx: IncidentContext, service: str) -> float:
    """Share of slow traces during the incident where this service holds the
    largest slice of wall-clock self time, measured against its own baseline
    share so a naturally slow service (payment) does not win by default."""
    during = ctx.mean_value(service, "slow_trace_participation")
    base = ctx.baseline(service, "slow_trace_participation")
    lift = during - base
    return float(np.clip(0.5 * during + 0.5 * max(0.0, lift) * 2.0, 0.0, 1.0))


def _self_latency_share(ctx: IncidentContext, service: str) -> float:
    """Of the latency this service gained, how much is its own work?

    ~1.0 means the service itself got slower (a cause). ~0.0 means all the
    extra time was spent waiting on a dependency (a symptom)."""
    d_total = ctx.delta(service, "latency_p95")
    d_self = ctx.delta(service, "self_latency_p95")
    if d_total <= EPS:
        # No latency change at all: neither evidence for nor against.
        return 0.5
    return float(np.clip(d_self / d_total, 0.0, 1.0))


def _error_originated(ctx: IncidentContext, service: str) -> float:
    """Errors this service produced itself rather than inherited as timeouts."""
    d_err = ctx.delta(service, "error_rate")
    if d_err <= 0.002:
        return 0.0
    d_timeout = ctx.delta(service, "timeout_rate")
    return float(np.clip((d_err - d_timeout) / (d_err + EPS), 0.0, 1.0))


def _deployment_evidence(ctx: IncidentContext, service: str, onset_time: float) -> float:
    dep = ctx.last_deployment_before(service, onset_time)
    if dep is None:
        return 0.0
    dt = onset_time - dep.timestamp
    r = SETTINGS.rca.deployment_recency_seconds
    if dt <= r:
        return 1.0
    if dt <= 2 * r:
        return float(1.0 - (dt - r) / r)
    return 0.0


def _log_evidence(ctx: IncidentContext, state: ServiceIncidentState) -> float:
    """How loudly this service complained, in its own logs."""
    z = state.peak_z.get("log_error_count", 0.0)
    unique = ctx.peak_value(state.service, "unique_error_types")
    return float(np.clip(0.7 * (z / 8.0) + 0.3 * min(1.0, unique / 3.0), 0.0, 1.0))


def build_candidates(ctx: IncidentContext) -> List[Dict[str, float]]:
    """One feature dict per candidate service."""
    states = ctx.states
    confirmed = ctx.confirmed
    if not states:
        return []
    onsets = {s: st.onset_window for s, st in states.items()}
    first, last = min(onsets.values()), max(onsets.values())
    span = max(1, last - first)
    max_depth = max(topo.distance_from_entrypoint(s) for s in topo.SERVICES) or 1
    n_others = max(1, len(confirmed) - 1)

    rows: List[Dict[str, float]] = []
    for service, st in states.items():
        prop, supporting, contradicting, deps_before = _propagation_consistency(
            service, states, confirmed
        )
        others_onsets = [onsets[s] for s in confirmed if s != service and s in onsets]
        lead_windows = (
            float(np.median(others_onsets)) - onsets[service] if others_onsets else 0.0
        )
        affected_callers = len(topo.transitive_callers(service) & confirmed)
        own_callees = topo.transitive_callees(service)
        callee_before = sum(
            1 for s2 in confirmed if s2 in own_callees and onsets.get(s2, last) < onsets[service]
        )

        features = {
            "anomaly_strength": float(np.clip(st.peak_score, 0.0, 1.0)),
            "anomaly_duration": float(np.clip(st.anomalous_fraction, 0.0, 1.0)),
            "early_onset": float(1.0 - (onsets[service] - first) / span),
            "onset_lead_seconds_norm": float(
                np.clip(lead_windows * ctx.window_seconds / 120.0, -1.0, 1.0)
            ),
            "propagation_consistency": prop,
            "downstream_affected_ratio": float(affected_callers / n_others),
            "dependency_broke_first": (
                float(callee_before / max(1, len(own_callees & confirmed)))
                if own_callees & confirmed
                else 0.0
            ),
            "self_latency_share": _self_latency_share(ctx, service),
            "error_originated": _error_originated(ctx, service),
            "trace_attribution": _trace_attribution(ctx, service),
            "deployment_evidence": _deployment_evidence(ctx, service, st.onset_time),
            "log_evidence": _log_evidence(ctx, st),
            "graph_depth_norm": float(topo.distance_from_entrypoint(service) / max_depth),
            "is_leaf": 1.0 if not topo.callees(service) else 0.0,
        }
        features["_supporting"] = float(supporting)
        features["_contradicting"] = float(contradicting)
        features["_deps_before"] = float(deps_before)
        features["_service"] = service  # type: ignore[assignment]
        rows.append(features)
    return rows


def to_matrix(rows: List[Dict[str, float]]) -> np.ndarray:
    return np.array(
        [[float(r.get(c, 0.0)) for c in CANDIDATE_FEATURES] for r in rows], dtype=float
    )
