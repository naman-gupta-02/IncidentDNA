"""Shared plumbing for stage one: is something wrong?

All detectors score the same object — one `(service, window)`, wrapped in an
`Observation` — and return a number in [0, 1]. Scores are comparable across
detectors so the ensemble can blend them, and the evidence carried on the
`Observation` is what the dashboard shows and what the RCA stage reuses.

Two design decisions live here:

**Detection is done on relative change, not absolute level.** Every detector
consumes the observation's vector, which expresses each watched metric as a
robust z-score and a log-ratio against that service's own recent baseline.
Feeding absolute values to an Isolation Forest makes it flag traffic spikes,
which are not incidents; feeding it relative change makes it load-invariant.

**Baselines are computed once per observation.** Building an `Observation`
costs one baseline lookup per metric; every detector then reads the cached
result instead of re-deriving it, which is what keeps a full replay of the
dataset in the tens of seconds rather than tens of minutes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import DETECTION_METRICS, SETTINGS
from ..pipeline.features import ServiceWindow
from ..pipeline.store import FeatureStore

#: Metrics that describe how much work arrived, not how well it went. They are
#: never evidence of a fault on their own, but a latency rise that is matched
#: by a load rise is explained by queueing rather than by a failure.
LOAD_METRICS: List[str] = ["request_count", "concurrent_requests"]

#: Metrics whose z-score should be discounted when load also rose.
LOAD_SENSITIVE_METRICS = {
    "latency_p95",
    "latency_p99",
    "self_latency_p95",
    "db_latency_p95",
    "cpu_mean",
}

#: Names of the columns in `Observation.vector`.
DETECTION_VECTOR_COLUMNS: List[str] = (
    [f"z__{m}" for m in DETECTION_METRICS] + [f"lr__{m}" for m in DETECTION_METRICS]
)


def _metric_scale_floor(metric: str) -> float:
    """Smallest deviation worth calling abnormal, in the metric's own units.

    Without this, a metric that is exactly constant during the baseline
    (error_rate == 0, restart_count == 0) produces an unbounded z the moment
    it moves by a rounding error."""
    return {
        "error_rate": 0.004,
        "timeout_rate": 0.004,
        "retry_rate": 0.004,
        "latency_p95": 4.0,
        "latency_p99": 6.0,
        "self_latency_p95": 3.0,
        "db_latency_p95": 3.0,
        "cpu_mean": 0.012,
        "memory_growth_rate": 0.6,
        "pool_utilization": 0.02,
        "log_error_count": 1.0,
        "restart_count": 0.5,
        "request_count": 8.0,
        "concurrent_requests": 0.5,
    }.get(metric, 1e-6)


def _z(value: float, median: float, scale: float, metric: str, one_sided: bool) -> float:
    scale = max(scale, abs(median) * 0.05, _metric_scale_floor(metric))
    z = (value - median) / scale
    if one_sided:
        z = max(0.0, z)
    return float(np.clip(z, -SETTINGS.detection.z_clip, SETTINGS.detection.z_clip))


@dataclass(slots=True)
class Observation:
    """One service in one window, with its baselines already resolved."""

    window: ServiceWindow
    #: Relative-change vector, or None when history is too short to judge.
    vector: Optional[np.ndarray]
    #: Raw one-sided z per detection metric.
    z: Dict[str, float] = field(default_factory=dict)
    #: Z per load metric (signed: load can legitimately fall).
    load_z: Dict[str, float] = field(default_factory=dict)

    @property
    def service(self) -> str:
        return self.window.service

    @property
    def ready(self) -> bool:
        return self.vector is not None

    @property
    def load_rise(self) -> float:
        """How much more work arrived than usual, in z units (never negative)."""
        return max(0.0, max(self.load_z.values(), default=0.0))

    def adjusted_z(self, metric: str, alpha: float) -> float:
        """Z with the part of the move that concurrent load explains removed."""
        z = self.z.get(metric, 0.0)
        if alpha <= 0 or metric not in LOAD_SENSITIVE_METRICS:
            return z
        return max(0.0, z - alpha * self.load_rise)


def observe(window: ServiceWindow, store: FeatureStore) -> Observation:
    """Resolve every baseline this window needs, exactly once."""
    zs: Dict[str, float] = {}
    ratios: List[float] = []
    for m in DETECTION_METRICS:
        base = store.robust_baseline(window.service, m, window.window_index)
        if base is None:
            return Observation(window=window, vector=None)
        median, scale, _ = base
        value = window.get(m)
        zs[m] = _z(value, median, scale, m, one_sided=True)
        ratios.append(math.log1p(max(0.0, value)) - math.log1p(max(0.0, median)))

    load_z: Dict[str, float] = {}
    for m in LOAD_METRICS:
        base = store.robust_baseline(window.service, m, window.window_index)
        if base is None:
            load_z[m] = 0.0
            continue
        median, scale, _ = base
        load_z[m] = _z(window.get(m), median, scale, m, one_sided=False)

    vec = np.array([zs[m] for m in DETECTION_METRICS] + ratios, dtype=float)
    return Observation(window=window, vector=vec, z=zs, load_z=load_z)


def robust_z(
    window: ServiceWindow, store: FeatureStore, metric: str, one_sided: bool = True
) -> Optional[float]:
    """Single-metric z-score. Convenience for tests and ad-hoc analysis;
    the detectors use `observe` so baselines are shared."""
    base = store.robust_baseline(window.service, metric, window.window_index)
    if base is None:
        return None
    median, scale, _ = base
    return _z(window.get(metric), median, scale, metric, one_sided)


def detection_vector(
    window: ServiceWindow, store: FeatureStore
) -> Tuple[Optional[np.ndarray], Dict[str, float]]:
    obs = observe(window, store)
    return obs.vector, obs.z


@dataclass
class ServiceAnomaly:
    """One service's verdict for one window."""

    service: str
    window_index: int
    window_start: float
    score: float
    is_anomalous: bool
    z_scores: Dict[str, float] = field(default_factory=dict)
    method_scores: Dict[str, float] = field(default_factory=dict)

    def top_metrics(self, n: int = 3) -> List[Tuple[str, float]]:
        return sorted(self.z_scores.items(), key=lambda kv: -kv[1])[:n]

    def to_dict(self) -> dict:
        return {
            "service": self.service,
            "window_index": self.window_index,
            "window_start": self.window_start,
            "score": round(self.score, 4),
            "is_anomalous": self.is_anomalous,
            "z_scores": {k: round(v, 3) for k, v in self.z_scores.items()},
            "method_scores": {k: round(v, 4) for k, v in self.method_scores.items()},
        }


class Detector:
    """Interface for a stage-one detector."""

    name = "detector"
    #: Whether the detector needs `fit` before it can score.
    requires_fit = False

    def fit(self, vectors: np.ndarray) -> "Detector":
        return self

    def score_batch(self, observations: Sequence[Observation]) -> List[Optional[float]]:
        """Score several observations at once. Learned detectors override this
        to make one vectorised call instead of one call per service."""
        return [self.score(o) for o in observations]

    def score(self, observation: Observation) -> Optional[float]:
        raise NotImplementedError


def logistic(x: float, midpoint: float, steepness: float) -> float:
    """Map an unbounded statistic onto (0, 1) with a known 0.5 crossing."""
    return 1.0 / (1.0 + math.exp(-(x - midpoint) / max(1e-6, steepness)))
