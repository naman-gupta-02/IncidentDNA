"""Blend the detectors into one score per (service, window)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..config import SETTINGS
from ..pipeline.features import ServiceWindow
from ..pipeline.store import FeatureStore
from .base import Detector, Observation, ServiceAnomaly, observe
from .detectors import (
    AutoencoderDetector,
    IsolationForestDetector,
    RollingZScoreDetector,
    StaticThresholdDetector,
)

DEFAULT_WEIGHTS: Dict[str, float] = {
    "rolling_zscore": 0.45,
    "static_threshold": 0.15,
    "isolation_forest": 0.20,
    "autoencoder": 0.20,
}


class EnsembleDetector:
    """Weighted blend, renormalised over whichever detectors can answer.

    Unfitted learned detectors simply abstain, so the same object works on a
    cold start (statistics only) and after training (statistics + ML).

    Scoring is done a whole window at a time — every service's observation
    together — so the learned detectors make one vectorised prediction per
    window instead of one per service.
    """

    def __init__(
        self,
        detectors: Optional[List[Detector]] = None,
        weights: Optional[Dict[str, float]] = None,
        threshold: Optional[float] = None,
    ) -> None:
        if detectors is None:
            detectors = [StaticThresholdDetector(), RollingZScoreDetector()]
        self.detectors = {d.name: d for d in detectors}
        self.weights = dict(weights or DEFAULT_WEIGHTS)
        self.threshold = (
            threshold if threshold is not None else SETTINGS.detection.service_score_threshold
        )

    # -- construction ------------------------------------------------------
    @classmethod
    def full(cls, random_state: int = 0, load_alpha: float = 0.6, **kwargs) -> "EnsembleDetector":
        return cls(
            detectors=[
                StaticThresholdDetector(),
                RollingZScoreDetector(load_alpha=load_alpha),
                IsolationForestDetector(random_state=random_state),
                AutoencoderDetector(random_state=random_state),
            ],
            **kwargs,
        )

    @classmethod
    def statistical_only(cls, load_alpha: float = 0.6, **kwargs) -> "EnsembleDetector":
        """The no-training baseline the ML detectors have to beat."""
        return cls(
            detectors=[StaticThresholdDetector(), RollingZScoreDetector(load_alpha=load_alpha)],
            **kwargs,
        )

    def fit(self, normal_vectors: np.ndarray) -> "EnsembleDetector":
        for d in self.detectors.values():
            if d.requires_fit:
                d.fit(normal_vectors)
        return self

    @property
    def fitted_detectors(self) -> List[str]:
        return [n for n, d in self.detectors.items() if not d.requires_fit or getattr(d, "fitted", False)]

    # -- scoring -----------------------------------------------------------
    def score_group(
        self, windows: Sequence[ServiceWindow], store: FeatureStore
    ) -> List[ServiceAnomaly]:
        """Score every service in one window together."""
        observations = [observe(w, store) for w in windows]
        per_method: Dict[str, List[Optional[float]]] = {
            name: d.score_batch(observations) for name, d in self.detectors.items()
        }
        out: List[ServiceAnomaly] = []
        for i, obs in enumerate(observations):
            method_scores = {
                name: float(vals[i]) for name, vals in per_method.items() if vals[i] is not None
            }
            total_w = sum(self.weights.get(n, 0.0) for n in method_scores)
            combined = (
                sum(self.weights.get(n, 0.0) * s for n, s in method_scores.items()) / total_w
                if total_w > 0
                else 0.0
            )
            out.append(
                ServiceAnomaly(
                    service=obs.service,
                    window_index=obs.window.window_index,
                    window_start=obs.window.window_start,
                    score=float(combined),
                    is_anomalous=bool(combined >= self.threshold),
                    z_scores=obs.z,
                    method_scores=method_scores,
                )
            )
        return out

    def score_window(self, window: ServiceWindow, store: FeatureStore) -> ServiceAnomaly:
        return self.score_group([window], store)[0]

    # -- persistence -------------------------------------------------------
    def save(self, path: str | Path) -> None:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"detectors": self.detectors, "weights": self.weights, "threshold": self.threshold},
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "EnsembleDetector":
        import joblib

        blob = joblib.load(path)
        obj = cls(detectors=list(blob["detectors"].values()), weights=blob["weights"])
        obj.threshold = blob["threshold"]
        return obj
