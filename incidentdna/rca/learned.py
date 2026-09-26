"""Learned root-cause rankers.

Each candidate service inside each incident becomes one supervised row with
label `is_root_cause`. That is a pointwise learning-to-rank setup: the model
scores rows independently and the ranking comes from sorting within an
incident, which is what `top-1 / top-3 / MRR` then measure.

Two models are provided — a logistic regression (a linear model whose
coefficients can be read next to the transparent baseline's hand weights) and
a gradient-boosted tree ensemble (the non-linear comparison the guide asks
for). Both are trained on whole runs held out from evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .candidates import CANDIDATE_FEATURES, to_matrix


@dataclass
class RankerTrainingData:
    """Rows grouped by incident."""

    features: np.ndarray          # (n_rows, n_features)
    labels: np.ndarray            # (n_rows,) 1 = true root cause
    groups: np.ndarray            # (n_rows,) incident id per row
    services: np.ndarray          # (n_rows,) candidate service per row

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def n_incidents(self) -> int:
        return len(np.unique(self.groups))


class LearnedRanker:
    """Wraps a sklearn classifier as a pointwise ranker."""

    def __init__(self, model_type: str = "logistic", random_state: int = 0) -> None:
        self.model_type = model_type
        self.random_state = random_state
        self.model = self._build(model_type, random_state)
        self.scaler = None
        self.fitted = False

    @property
    def name(self) -> str:
        return f"learned_{self.model_type}"

    @staticmethod
    def _build(model_type: str, random_state: int):
        if model_type == "logistic":
            from sklearn.linear_model import LogisticRegression

            return LogisticRegression(
                max_iter=2000, C=1.0, class_weight="balanced", random_state=random_state
            )
        if model_type == "gbm":
            from sklearn.ensemble import HistGradientBoostingClassifier

            return HistGradientBoostingClassifier(
                max_iter=250,
                learning_rate=0.06,
                max_depth=4,
                min_samples_leaf=15,
                l2_regularization=1.0,
                random_state=random_state,
            )
        if model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier

            return RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
            )
        raise ValueError(f"unknown ranker model '{model_type}'")

    # -- training ----------------------------------------------------------
    def fit(self, data: RankerTrainingData) -> "LearnedRanker":
        x = data.features
        if self.model_type == "logistic":
            from sklearn.preprocessing import StandardScaler

            self.scaler = StandardScaler()
            x = self.scaler.fit_transform(x)
        self.model.fit(x, data.labels)
        self.fitted = True
        return self

    def _prepare(self, x: np.ndarray) -> np.ndarray:
        return self.scaler.transform(x) if self.scaler is not None else x

    def score_matrix(self, x: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError(f"{self.name} is not fitted")
        return self.model.predict_proba(self._prepare(x))[:, 1]

    def score(self, rows: Sequence[Dict[str, float]]) -> np.ndarray:
        if not rows:
            return np.array([])
        return self.score_matrix(to_matrix(list(rows)))

    # -- introspection -----------------------------------------------------
    def feature_importance(self) -> Dict[str, float]:
        if not self.fitted:
            return {}
        if hasattr(self.model, "coef_"):
            coefs = self.model.coef_[0]
            return {f: float(c) for f, c in zip(CANDIDATE_FEATURES, coefs)}
        if hasattr(self.model, "feature_importances_"):
            return {
                f: float(c)
                for f, c in zip(CANDIDATE_FEATURES, self.model.feature_importances_)
            }
        # HistGradientBoosting has no native importances; use permutation on
        # the training set at evaluation time instead.
        return {}

    def contributions(self, row: Dict[str, float]) -> Dict[str, float]:
        """Per-feature contribution for the dashboard. Exact for the linear
        model; for trees we fall back to the standardised feature value times
        its global importance, which is indicative rather than exact."""
        if not self.fitted:
            return {}
        x = to_matrix([row])
        if self.scaler is not None and hasattr(self.model, "coef_"):
            xs = self.scaler.transform(x)[0]
            return {
                f: round(float(xs[i] * self.model.coef_[0][i]), 4)
                for i, f in enumerate(CANDIDATE_FEATURES)
            }
        imp = self.feature_importance()
        return {f: round(float(row.get(f, 0.0)) * imp.get(f, 0.0), 4) for f in CANDIDATE_FEATURES}

    # -- persistence -------------------------------------------------------
    def save(self, path: str | Path) -> None:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model_type": self.model_type,
                "model": self.model,
                "scaler": self.scaler,
                "features": CANDIDATE_FEATURES,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "LearnedRanker":
        import joblib

        blob = joblib.load(path)
        obj = cls(model_type=blob["model_type"])
        obj.model = blob["model"]
        obj.scaler = blob["scaler"]
        obj.fitted = True
        if blob.get("features") != CANDIDATE_FEATURES:
            raise ValueError(
                "saved ranker was trained on a different candidate feature set; retrain it"
            )
        return obj


def build_training_data(
    rows: Sequence[Dict[str, float]],
    incident_ids: Sequence[str],
    root_causes: Sequence[str],
) -> RankerTrainingData:
    """`rows[i]` belongs to incident `incident_ids[i]` whose true origin is
    `root_causes[i]`."""
    features = to_matrix(list(rows))
    services = np.array([r["_service"] for r in rows])
    labels = np.array(
        [1 if r["_service"] == rc else 0 for r, rc in zip(rows, root_causes)], dtype=int
    )
    return RankerTrainingData(features, labels, np.array(incident_ids), services)
