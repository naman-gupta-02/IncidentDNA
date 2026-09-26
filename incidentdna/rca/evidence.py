"""Turn a candidate's feature row back into sentences a human can check.

Every sentence here is generated from a stored measurement or event, and each
one names the numbers it came from. Nothing is invented: if the measurement
does not clear the bar for being interesting, the sentence is simply not
emitted. An LLM could later rewrite the phrasing, but it must not be allowed
to supply the facts.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .. import topology as topo
from ..incidents.models import ServiceIncidentState
from .candidates import IncidentContext

#: Metrics quoted in evidence, in the order they are preferred, with the unit
#: formatter to use.
_QUOTABLE: List[Tuple[str, str, str]] = [
    ("db_latency_p95", "database query latency (p95)", "ms"),
    ("self_latency_p95", "own processing latency (p95)", "ms"),
    ("latency_p95", "request latency (p95)", "ms"),
    ("pool_utilization", "connection-pool utilisation", "pct"),
    ("pool_wait_ms", "connection-pool wait time", "ms"),
    ("error_rate", "error rate", "pct"),
    ("timeout_rate", "timeout rate", "pct"),
    ("memory_growth_rate", "memory growth", "mbmin"),
    ("cpu_mean", "CPU utilisation", "pct"),
    ("restart_count", "process restarts", "count"),
    ("log_error_count", "error-log volume", "count"),
]


def fmt(value: float, unit: str) -> str:
    if unit == "ms":
        return f"{value / 1000.0:.2f} s" if value >= 1000 else f"{value:.0f} ms"
    if unit == "pct":
        return f"{value * 100:.0f}%"
    if unit == "mbmin":
        return f"{value:.1f} MB/min"
    return f"{value:.0f}"


def _name(service: str) -> str:
    return topo.DISPLAY_NAMES.get(service, service)


def _significant_metric_changes(
    ctx: IncidentContext, service: str, limit: int = 3
) -> List[Tuple[str, str, float, float]]:
    """(label, unit, baseline, peak) for metrics that moved materially."""
    out = []
    for metric, label, unit in _QUOTABLE:
        base = ctx.baseline(service, metric)
        peak = ctx.peak_value(service, metric)
        if peak <= 0:
            continue
        moved = (peak > base * 1.8 and peak - base > _floor(metric)) or (
            base == 0 and peak > _floor(metric)
        )
        if moved:
            out.append((label, unit, base, peak))
        if len(out) >= limit:
            break
    return out


def _floor(metric: str) -> float:
    return {
        "db_latency_p95": 25.0,
        "self_latency_p95": 25.0,
        "latency_p95": 40.0,
        "pool_utilization": 0.15,
        "pool_wait_ms": 40.0,
        "error_rate": 0.01,
        "timeout_rate": 0.01,
        "memory_growth_rate": 4.0,
        "cpu_mean": 0.12,
        "restart_count": 0.5,
        "log_error_count": 4.0,
    }.get(metric, 0.0)


def propagation_path(ctx: IncidentContext) -> List[str]:
    """Affected services in onset order, keeping only graph-adjacent hops so
    the arrow chain describes a real call path."""
    states = ctx.states
    ordered = sorted(
        (s for s in states if s in ctx.confirmed), key=lambda s: states[s].onset_window
    )
    path = [ordered[0]] if ordered else []
    for s in ordered[1:]:
        if any(prev in topo.transitive_callees(s) for prev in path):
            path.append(s)
    return path


def build_evidence(
    ctx: IncidentContext, row: Dict[str, float], state: ServiceIncidentState
) -> Tuple[List[str], List[str]]:
    """(supporting, contradicting) sentences for one candidate."""
    service = state.service
    name = _name(service)
    states = ctx.states
    support: List[str] = []
    against: List[str] = []
    W = ctx.window_seconds

    # 1. What changed, in units.
    changes = _significant_metric_changes(ctx, service)
    for label, unit, base, peak in changes[:2]:
        support.append(
            f"{name} {label} rose from {fmt(base, unit)} to {fmt(peak, unit)}."
        )

    # 2. Onset lead over the services that call it.
    callers_affected = [
        s for s in ctx.confirmed
        if s != service and s in states and s in topo.transitive_callers(service)
    ]
    later = [s for s in callers_affected if states[s].onset_window > state.onset_window]
    if later:
        nearest = min(later, key=lambda s: states[s].onset_window)
        lead = (states[nearest].onset_window - state.onset_window) * W
        support.append(
            f"{name} became abnormal {lead:.0f} s before {_name(nearest)}, which calls it."
        )
    elif callers_affected:
        earliest = min(callers_affected, key=lambda s: states[s].onset_window)
        if states[earliest].onset_window < state.onset_window:
            gap = (state.onset_window - states[earliest].onset_window) * W
            against.append(
                f"{_name(earliest)} calls {name} but became abnormal {gap:.0f} s earlier, "
                f"which is the wrong order for {name} to be the origin."
            )

    # 3. Where the extra time actually went.
    share = row.get("self_latency_share", 0.5)
    d_total = ctx.delta(service, "latency_p95")
    if d_total > 30:
        if share >= 0.6:
            support.append(
                f"{share * 100:.0f}% of {name}'s added latency was its own work, "
                f"not time spent waiting on dependencies."
            )
        elif share <= 0.35:
            against.append(
                f"Only {share * 100:.0f}% of {name}'s added latency was its own work — "
                f"the rest was spent waiting on a dependency."
            )

    # 4. Trace attribution.
    attribution = ctx.mean_value(service, "slow_trace_participation")
    if attribution >= 0.3:
        support.append(
            f"{name} held the largest share of request time in "
            f"{attribution * 100:.0f}% of slow traces during the incident."
        )

    # 5. Deployment proximity.
    dep = ctx.last_deployment_before(service, state.onset_time)
    if dep is not None and row.get("deployment_evidence", 0.0) > 0:
        dt = state.onset_time - dep.timestamp
        support.append(
            f"{name} was deployed ({dep.previous_version} to {dep.version}) "
            f"{dt:.0f} s before it became abnormal."
        )
    others_deployed = [
        s
        for s in ctx.confirmed
        if s != service and s in states
        and ctx.last_deployment_before(s, states[s].onset_time) is not None
        and states[s].onset_time - ctx.last_deployment_before(s, states[s].onset_time).timestamp
        < 2 * 60
    ]
    if dep is not None and row.get("deployment_evidence", 0.0) > 0 and not others_deployed:
        support.append("No deployment occurred in the other affected services.")

    # 6. Errors generated here rather than inherited.
    if row.get("error_originated", 0.0) >= 0.6 and ctx.delta(service, "error_rate") > 0.01:
        support.append(
            f"{name}'s errors were generated locally: its error rate rose without a "
            f"matching rise in timeouts on its outbound calls."
        )

    # 7. Propagation chain.
    path = propagation_path(ctx)
    if len(path) >= 2 and path[0] == service:
        support.append(
            "The anomaly propagated " + " then ".join(_name(s) for s in path) + "."
        )

    # 8. Contradiction: one of its own dependencies broke first.
    own_callees = topo.transitive_callees(service) & set(ctx.confirmed)
    earlier_deps = [s for s in own_callees if states[s].onset_window < state.onset_window]
    if earlier_deps:
        first = min(earlier_deps, key=lambda s: states[s].onset_window)
        gap = (state.onset_window - states[first].onset_window) * W
        against.append(
            f"{name} depends on {_name(first)}, which became abnormal {gap:.0f} s earlier."
        )

    if not support:
        support.append(
            f"{name} was flagged abnormal for {state.anomalous_windows} windows "
            f"(peak score {state.peak_score:.2f}), but no single metric moved decisively."
        )
    return support, against


def build_symptom_text(ctx: IncidentContext) -> Optional[str]:
    """Describe the user-visible symptom, preferring the entrypoint service.

    Only services that actually tripped the alert can be the symptom — a quiet
    dependency pulled into the candidate pool is a possible cause, never the
    thing a user noticed."""
    states = ctx.states
    confirmed = sorted(ctx.confirmed, key=lambda s: -states[s].peak_score)
    for service in list(topo.ENTRYPOINTS) + confirmed:
        if service not in states or service not in ctx.confirmed:
            continue
        changes = _significant_metric_changes(ctx, service, limit=1)
        if changes:
            label, unit, base, peak = changes[0]
            return f"{_name(service)} {label} rose from {fmt(base, unit)} to {fmt(peak, unit)}"
    return None
