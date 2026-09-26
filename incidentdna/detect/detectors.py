"""The four stage-one detectors from the guide.

| detector                | labels needed  | catches                              |
|-------------------------|----------------|--------------------------------------|
| StaticThreshold         | none           | outright breaches (5xx, pool at 99%) |
| RollingZScore           | none           | deviation from a service's own norm  |
| IsolationForestDetector | normal windows | odd *combinations* of deviations     |
| AutoencoderDetector     | normal windows | broken relationships between metrics |

The two learned detectors are fitted only on windows drawn from runs with no
injected fault, so "normal" never contains an incident.

`RollingZScoreDetector` additionally supports **load adjustment**: latency and
CPU rise whenever traffic rises, purely from queueing, so a deviation that is
proportional to a concurrent load rise is discounted. Traffic spikes are the
dominant source of false alerts without it, and `scripts/evaluate.py` reports
the detection metrics with it on and off.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..config import DETECTION_METRICS, SETTINGS
from .base import Detector, Observation, logistic

_D = SETTINGS.detection


class StaticThresholdDetector(Detector):
    """Operational rules an on-call engineer would write by hand.

    Fast and completely explainable, but blind to anything that is bad *for
    this service* while still under a global threshold."""

    name = "static_threshold"

    DEFAULT_RULES: Dict[str, float] = {
        "error_rate": 0.05,
        "timeout_rate": 0.03,
        "latency_p95": 1500.0,
        "cpu_mean": 0.90,
        "pool_utilization": 0.90,
        "memory_growth_rate": 30.0,
        "restart_count": 1.0,
    }

    def __init__(self, rules: Optional[Dict[str, float]] = None) -> None:
        self.rules = dict(rules or self.DEFAULT_RULES)

    def score(self, observation: Observation) -> Optional[float]:
        window = observation.window
        breaches = [
            min(3.0, window.get(metric) / limit)
            for metric, limit in self.rules.items()
            if limit > 0 and window.get(metric) > limit
        ]
        if not breaches:
            return 0.0
        worst = max(breaches)
        # One breach is already actionable; several are worse.
        return float(min(1.0, 0.55 + 0.15 * (worst - 1.0) + 0.1 * (len(breaches) - 1)))

    def breached(self, observation: Observation) -> List[str]:
        w = observation.window
        return [m for m, lim in self.rules.items() if lim > 0 and w.get(m) > lim]


class RollingZScoreDetector(Detector):
    """Median/MAD deviation from the service's own recent behaviour.

    Metrics are combined with a top-k RMS so several mildly abnormal metrics
    outrank one noisy metric — a single jumpy p99 should not page anyone.
    """

    name = "rolling_zscore"

    def __init__(self, top_k: int = 3, load_alpha: float = 0.6) -> None:
        self.top_k = top_k
        #: 0 disables load adjustment; 0.6 removes most of the queueing effect
        #: of a traffic spike while leaving a genuine latency fault intact.
        self.load_alpha = load_alpha

    def combined_z(self, observation: Observation) -> Optional[float]:
        if not observation.ready:
            return None
        zs = [observation.adjusted_z(m, self.load_alpha) for m in DETECTION_METRICS]
        top = sorted(zs, reverse=True)[: self.top_k]
        if not top:
            return None
        return float(math.sqrt(sum(z * z for z in top) / len(top)))

    def score(self, observation: Observation) -> Optional[float]:
        z = self.combined_z(observation)
        if z is None:
            return None
        return logistic(z, midpoint=_D.z_alert, steepness=1.1)


class _BatchedModelDetector(Detector):
    """Shared calibration and batching for the two learned detectors.

    Raw scores are mapped through a logistic calibrated on the *training*
    distribution so that the 99.5th percentile of normal lands near 0.9,
    which puts every detector on the same 0-1 scale as the statistical ones.
    """

    requires_fit = True

    def __init__(self) -> None:
        self.fitted = False
        self._ref_median = 0.0
        self._ref_scale = 1.0

    def _calibrate(self, raw: np.ndarray) -> None:
        self._ref_median = float(np.median(raw))
        hi = float(np.percentile(raw, 99.5))
        self._ref_scale = max(1e-6, (hi - self._ref_median) / 2.2)

    def _to_score(self, raw: float) -> float:
        return logistic(
            raw, midpoint=self._ref_median + self._ref_scale, steepness=self._ref_scale
        )

    def _raw_batch(self, matrix: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def score_batch(self, observations: Sequence[Observation]) -> List[Optional[float]]:
        out: List[Optional[float]] = [None] * len(observations)
        if not self.fitted:
            return out
        ready = [i for i, o in enumerate(observations) if o.ready]
        if not ready:
            return out
        matrix = np.vstack([observations[i].vector for i in ready])
        raw = self._raw_batch(matrix)
        for slot, i in enumerate(ready):
            out[i] = self._to_score(float(raw[slot]))
        return out

    def score(self, observation: Observation) -> Optional[float]:
        return self.score_batch([observation])[0]


class IsolationForestDetector(_BatchedModelDetector):
    """Multivariate outlier scoring over the relative-change vector."""

    name = "isolation_forest"

    def __init__(self, n_estimators: int = 150, random_state: int = 0) -> None:
        super().__init__()
        from sklearn.ensemble import IsolationForest

        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination="auto",
            random_state=random_state,
            n_jobs=1,
        )

    def fit(self, vectors: np.ndarray) -> "IsolationForestDetector":
        if len(vectors) < 50:
            raise ValueError("need at least 50 normal windows to fit IsolationForest")
        self.model.fit(vectors)
        self._calibrate(-self.model.score_samples(vectors))
        self.fitted = True
        return self

    def _raw_batch(self, matrix: np.ndarray) -> np.ndarray:
        return -self.model.score_samples(matrix)


class AutoencoderDetector(_BatchedModelDetector):
    """Reconstruction-error detector.

    A bottlenecked MLP is trained to reproduce normal relative-change vectors.
    When metrics move in a combination the model has never seen — latency up
    while CPU is flat and the pool is saturated — reconstruction error jumps.
    Implemented with scikit-learn so the project needs no deep-learning stack.
    """

    name = "autoencoder"

    def __init__(
        self, hidden: tuple = (16, 6, 16), random_state: int = 0, max_iter: int = 400
    ) -> None:
        super().__init__()
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler

        self.scaler = StandardScaler()
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden,
            activation="tanh",
            solver="adam",
            learning_rate_init=3e-3,
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=True,
            n_iter_no_change=15,
        )

    def fit(self, vectors: np.ndarray) -> "AutoencoderDetector":
        if len(vectors) < 100:
            raise ValueError("need at least 100 normal windows to fit the autoencoder")
        x = self.scaler.fit_transform(vectors)
        self.model.fit(x, x)
        self._calibrate(self._errors(x))
        self.fitted = True
        return self

    def _errors(self, x: np.ndarray) -> np.ndarray:
        return np.mean((x - self.model.predict(x)) ** 2, axis=1)

    def _raw_batch(self, matrix: np.ndarray) -> np.ndarray:
        return self._errors(self.scaler.transform(matrix))

    def reconstruction_error(self, observation: Observation) -> Optional[float]:
        if not self.fitted or not observation.ready:
            return None
        return float(self._raw_batch(observation.vector.reshape(1, -1))[0])
