"""Catalogue of injectable faults.

Each fault knows which services it can originate in and how it distorts that
service's behaviour over time. Everything downstream of the origin — slower
callers, timeouts, retries, gateway errors — is *not* written here: it falls
out of the dependency graph in `engine.py`. That separation is what makes the
generated incidents honest: the simulator never labels a propagation path, it
only labels the origin.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .. import topology as topo


@dataclass
class FaultContext:
    """Where we are inside an injected fault."""

    progress: float          # 0..1 across the fault's lifetime
    elapsed_seconds: float
    severity: float          # 0..1, sampled per run
    rng: "object"            # numpy Generator
    state: Dict[str, float] = field(default_factory=dict)  # per-run scratch


@dataclass
class FaultEffect:
    """How the origin service misbehaves this window."""

    self_latency_mult: float = 1.0
    self_latency_add_ms: float = 0.0
    self_latency_sigma_add: float = 0.0
    cpu_add: float = 0.0
    memory_add_mb: float = 0.0
    pool_utilization: Optional[float] = None
    pool_wait_ms: float = 0.0
    intrinsic_error_rate: float = 0.0
    capacity_mult: float = 1.0
    restart: bool = False
    logs: List[Tuple[str, str, int]] = field(default_factory=list)  # severity, event, count
    attributes: Dict[str, object] = field(default_factory=dict)
    deploy_version: Optional[str] = None


def ramp(progress: float, rise: float = 0.18, fall: float = 0.12) -> float:
    """Trapezoid: degrade over `rise`, hold, recover over `fall`.

    Real incidents rarely step instantly to their worst value, and a ramp is
    what makes anomaly-onset ordering a genuinely noisy signal rather than a
    clean step the detector can trivially rank."""
    if progress <= 0.0:
        return 0.0
    if progress >= 1.0:
        return 0.0
    if progress < rise:
        return progress / rise
    if progress > 1.0 - fall:
        return max(0.0, (1.0 - progress) / fall)
    return 1.0


def saturating(progress: float, k: float = 6.0) -> float:
    """Monotone rise that never recovers within the window (memory leaks,
    connection pools) — recovery only happens on restart."""
    return 1.0 - math.exp(-k * max(0.0, progress))


@dataclass
class FaultSpec:
    name: str
    description: str
    origin_candidates: List[str]
    effect_fn: Callable[[FaultContext], FaultEffect]
    #: Injection mechanism, quoted in the UI and the README.
    mechanism: str = ""
    expected_propagation: str = ""

    def effect(self, ctx: FaultContext) -> FaultEffect:
        return self.effect_fn(ctx)


# --- individual faults -----------------------------------------------------
def _database_latency(ctx: FaultContext) -> FaultEffect:
    """A query plan regression: the index behind SELECT/INSERT orders is gone."""
    r = ramp(ctx.progress)
    mult = 1.0 + r * (4.0 + 26.0 * ctx.severity)
    return FaultEffect(
        self_latency_mult=mult,
        self_latency_sigma_add=0.25 * r,
        cpu_add=0.22 * r * ctx.severity,
        logs=[("WARN", "slow_query", int(4 + 30 * r * ctx.severity))],
        attributes={"query": "SELECT orders", "missing_index": "idx_orders_user_id"},
    )


def _connection_exhaustion(ctx: FaultContext) -> FaultEffect:
    """Pool sized too small for the offered concurrency: waiters queue up."""
    s = saturating(ctx.progress * 1.6) * ramp(ctx.progress, rise=0.05, fall=0.15)
    util = 0.42 + s * (0.5 + 0.06 * ctx.severity)
    wait = s * (180.0 + 1400.0 * ctx.severity)
    return FaultEffect(
        self_latency_add_ms=wait,
        pool_utilization=min(0.995, util),
        pool_wait_ms=wait,
        logs=[("WARN", "connection_pool_saturated", int(2 + 18 * s))],
        attributes={"pool_size": 10, "waiters": int(s * 40)},
    )


def _memory_leak(ctx: FaultContext) -> FaultEffect:
    """Objects retained per request; GC pressure then an OOM restart."""
    leaked = ctx.state.get("leaked_mb", 0.0)
    rate = 1.6 + 7.0 * ctx.severity          # MB per window
    leaked += rate
    restart = False
    if leaked > 620.0:                        # container limit
        leaked = 0.0
        restart = True
        ctx.state["restarts"] = ctx.state.get("restarts", 0) + 1
    ctx.state["leaked_mb"] = leaked
    pressure = min(1.0, leaked / 620.0)
    logs: List[Tuple[str, str, int]] = []
    if pressure > 0.55:
        logs.append(("WARN", "gc_pressure", int(3 + 20 * pressure)))
    if restart:
        logs.append(("ERROR", "oom_killed", 1))
    return FaultEffect(
        memory_add_mb=leaked,
        cpu_add=0.35 * pressure ** 2,
        self_latency_mult=1.0 + 2.6 * pressure ** 2,
        intrinsic_error_rate=0.55 if restart else 0.0,
        restart=restart,
        logs=logs,
        attributes={"heap_mb": round(leaked, 1), "gc_pause_ratio": round(0.3 * pressure, 3)},
    )


def _payment_timeout(ctx: FaultContext) -> FaultEffect:
    """Upstream payment provider degrades; some authorisations never return."""
    r = ramp(ctx.progress, rise=0.08)
    mult = 1.0 + r * (3.0 + 14.0 * ctx.severity)
    return FaultEffect(
        self_latency_mult=mult,
        self_latency_sigma_add=0.45 * r,
        intrinsic_error_rate=r * 0.10 * ctx.severity,
        logs=[("ERROR", "provider_timeout", int(2 + 22 * r * ctx.severity))],
        attributes={"provider": "acme-psp", "circuit_breaker": r > 0.7},
    )


def _cpu_saturation(ctx: FaultContext) -> FaultEffect:
    """An expensive code path pins the workers; queueing does the rest."""
    r = ramp(ctx.progress, rise=0.1)
    return FaultEffect(
        cpu_add=r * (0.45 + 0.45 * ctx.severity),
        capacity_mult=1.0 - r * (0.35 + 0.45 * ctx.severity),
        self_latency_mult=1.0 + r * (0.6 + 1.4 * ctx.severity),
        logs=[("WARN", "worker_pool_exhausted", int(1 + 12 * r))],
        attributes={"hot_path": "price_recalculation"},
    )


def _bad_deployment(ctx: FaultContext) -> FaultEffect:
    """A release that fails for a subset of inputs: a step change in errors,
    anchored to a deployment event the ranker can use as evidence."""
    live = 1.0 if 0.02 < ctx.progress < 0.97 else 0.0
    return FaultEffect(
        intrinsic_error_rate=live * (0.04 + 0.30 * ctx.severity),
        self_latency_mult=1.0 + live * 0.35 * ctx.severity,
        logs=[("ERROR", "unhandled_exception", int(live * (3 + 26 * ctx.severity)))],
        attributes={"regression": "null_shipping_address", "rollout": "100%"},
        deploy_version="v2.0.0" if ctx.progress < 0.02 else None,
    )


FAULTS: Dict[str, FaultSpec] = {
    "database_latency": FaultSpec(
        name="database_latency",
        description="PostgreSQL query latency regression (missing index)",
        origin_candidates=[topo.POSTGRES],
        effect_fn=_database_latency,
        mechanism="Drop an index / inject a statement-level delay in PostgreSQL.",
        expected_propagation="DB latency -> Order latency -> Gateway latency -> timeouts.",
    ),
    "connection_exhaustion": FaultSpec(
        name="connection_exhaustion",
        description="Connection pool saturated, requests queue for a connection",
        origin_candidates=[topo.POSTGRES],
        effect_fn=_connection_exhaustion,
        mechanism="Shrink the pool while holding concurrent traffic steady.",
        expected_propagation="Pool saturation -> wait time -> DB duration -> request timeout.",
    ),
    "memory_leak": FaultSpec(
        name="memory_leak",
        description="Unbounded retention in the order worker, ending in an OOM restart",
        origin_candidates=[topo.ORDER_SERVICE],
        effect_fn=_memory_leak,
        mechanism="Retain request objects in a process-level list.",
        expected_propagation="Memory growth -> GC pressure -> latency -> restart -> errors.",
    ),
    "payment_timeout": FaultSpec(
        name="payment_timeout",
        description="Payment provider slow, authorisations time out and are retried",
        origin_candidates=[topo.PAYMENT_SERVICE],
        effect_fn=_payment_timeout,
        mechanism="Delay or drop payment responses (Toxiproxy latency toxic).",
        expected_propagation="Payment timeouts -> retries -> checkout latency and failures.",
    ),
    "cpu_saturation": FaultSpec(
        name="cpu_saturation",
        description="CPU pinned by an expensive computation, workers stall",
        origin_candidates=[topo.ORDER_SERVICE, topo.INVENTORY_SERVICE, topo.API_GATEWAY],
        effect_fn=_cpu_saturation,
        mechanism="Run a busy loop / stress-ng inside the container.",
        expected_propagation="CPU rises -> workers stall -> service latency and queue growth.",
    ),
    "bad_deployment": FaultSpec(
        name="bad_deployment",
        description="Release regression: a step change in errors for some inputs",
        origin_candidates=[topo.API_GATEWAY, topo.ORDER_SERVICE, topo.INVENTORY_SERVICE],
        effect_fn=_bad_deployment,
        mechanism="Ship a build that raises for selected inputs.",
        expected_propagation="Version change -> error-rate step change in that service.",
    ),
}

FAULT_NAMES: List[str] = list(FAULTS)


def get(name: str) -> FaultSpec:
    try:
        return FAULTS[name]
    except KeyError:
        raise KeyError(f"unknown fault '{name}'; known: {', '.join(FAULT_NAMES)}") from None


def catalogue() -> List[dict]:
    """Serialisable description for the dashboard's inject menu."""
    return [
        {
            "name": f.name,
            "description": f.description,
            "origins": f.origin_candidates,
            "mechanism": f.mechanism,
            "expected_propagation": f.expected_propagation,
        }
        for f in FAULTS.values()
    ]
