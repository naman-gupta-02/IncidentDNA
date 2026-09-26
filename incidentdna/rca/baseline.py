"""Transparent root-cause scoring.

Two scorers live here:

`TransparentScorer` implements the guide's formula verbatim ::

    root_score(v) = 0.35 * anomaly_strength
                  + 0.25 * early_onset
                  + 0.20 * propagation_consistency
                  + 0.10 * deployment_evidence
                  + 0.10 * log_and_trace_evidence

`CausalLocalityScorer` keeps that shape but adds the two terms that the
telemetry makes available and that most directly separate cause from symptom:
whether the service's *own* work slowed down, and whether its errors were
self-generated rather than inherited timeouts. It is still a hand-weighted
linear model with no training, so it stays inspectable — it exists so the
learned rankers are measured against a strong transparent baseline and not
only a weak one.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from ..config import SETTINGS
from .candidates import CANDIDATE_FEATURES

_W = SETTINGS.rca


class WeightedScorer:
    """Linear scorer over candidate features with per-term attribution."""

    name = "weighted"
    weights: Dict[str, float] = {}

    def score_row(self, row: Dict[str, float]) -> float:
        return float(sum(w * float(row.get(k, 0.0)) for k, w in self.weights.items()))

    def contributions(self, row: Dict[str, float]) -> Dict[str, float]:
        return {k: round(w * float(row.get(k, 0.0)), 4) for k, w in self.weights.items()}

    def score(self, rows: Sequence[Dict[str, float]]) -> np.ndarray:
        return np.array([self.score_row(r) for r in rows], dtype=float)

    def explain_weights(self) -> Dict[str, float]:
        return dict(self.weights)


class TransparentScorer(WeightedScorer):
    """The guide's baseline, unchanged."""

    name = "transparent"

    def __init__(self) -> None:
        self.weights = {
            "anomaly_strength": _W.w_anomaly_strength,
            "early_onset": _W.w_early_onset,
            "propagation_consistency": _W.w_propagation_consistency,
            "deployment_evidence": _W.w_deployment_evidence,
            "log_and_trace_evidence": _W.w_log_and_trace_evidence,
        }

    def score_row(self, row: Dict[str, float]) -> float:
        enriched = dict(row)
        # The guide bundles logs and traces into a single evidence term.
        enriched["log_and_trace_evidence"] = 0.5 * float(row.get("log_evidence", 0.0)) + 0.5 * float(
            row.get("trace_attribution", 0.0)
        )
        return super().score_row(enriched)

    def contributions(self, row: Dict[str, float]) -> Dict[str, float]:
        enriched = dict(row)
        enriched["log_and_trace_evidence"] = 0.5 * float(row.get("log_evidence", 0.0)) + 0.5 * float(
            row.get("trace_attribution", 0.0)
        )
        return super().contributions(enriched)


class CausalLocalityScorer(WeightedScorer):
    """Transparent, untrained, but aware of where the latency was actually
    spent and where the errors were actually produced.

    Two things distinguish it from the plain weighted sum:

    *A quiet service is not a cause.* Locality evidence — "its own work got
    slower", "it dominates slow traces" — is scaled by how far the service
    actually deviated. Without that gate an idle leaf dependency scores well
    on every locality term simply because it has no dependencies of its own to
    blame, and the wider candidate pool is full of those.

    *A service whose own dependency broke first is almost never the cause*,
    so that carries an explicit penalty rather than merely low weight.
    """

    name = "causal_locality"

    #: Below this anomaly strength a candidate's locality evidence is
    #: discounted proportionally; at or above it, it counts in full.
    GATE_AT = 0.55

    def __init__(self) -> None:
        self.weights = {
            "anomaly_strength": 0.30,
            "self_latency_share": 0.18,
            "early_onset": 0.15,
            "propagation_consistency": 0.13,
            "trace_attribution": 0.10,
            "error_originated": 0.07,
            "deployment_evidence": 0.04,
            "log_evidence": 0.03,
        }
        #: Terms that only mean something for a service that actually moved.
        self.gated = {
            "self_latency_share",
            "trace_attribution",
            "error_originated",
            "log_evidence",
        }

    def _gate(self, row: Dict[str, float]) -> float:
        return float(np.clip(float(row.get("anomaly_strength", 0.0)) / self.GATE_AT, 0.0, 1.0))

    def contributions(self, row: Dict[str, float]) -> Dict[str, float]:
        gate = self._gate(row)
        out = {
            k: round(w * float(row.get(k, 0.0)) * (gate if k in self.gated else 1.0), 4)
            for k, w in self.weights.items()
        }
        out["dependency_broke_first"] = round(
            -0.25 * float(row.get("dependency_broke_first", 0.0)), 4
        )
        return out

    def score_row(self, row: Dict[str, float]) -> float:
        return float(sum(self.contributions(row).values()))


class AnomalyStrengthScorer(WeightedScorer):
    """The naive thing everyone tries first: blame whatever looks worst.

    It is the control the whole project exists to beat. Amplification means
    the loudest service is frequently a symptom — a gateway that is timing out
    looks far sicker than the database that is making it time out."""

    name = "naive_anomaly_strength"

    def __init__(self) -> None:
        self.weights = {"anomaly_strength": 1.0}


class EarliestOnsetScorer(WeightedScorer):
    """The second thing everyone tries: blame whatever broke first.

    Strong, but it inherits the detector's timing noise, and a cause whose
    early symptoms are subtle can be detected *after* the service it breaks."""

    name = "naive_earliest_onset"

    def __init__(self) -> None:
        self.weights = {"early_onset": 1.0}


class RandomScorer:
    """Chance, given the candidate set actually produced. Establishes the
    floor that top-1 and MRR should be read against."""

    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def score(self, rows: Sequence[Dict[str, float]]) -> np.ndarray:
        return self.rng.random(len(rows))

    def contributions(self, row: Dict[str, float]) -> Dict[str, float]:
        return {}


def softmax_confidence(scores: np.ndarray, temperature: float = 0.22) -> np.ndarray:
    """Turn raw scores into a calibrated-looking confidence distribution.

    Temperature is set so a clear winner lands around 0.85-0.95 and a close
    two-way call lands near 0.55, which is what the dashboard should show."""
    if len(scores) == 0:
        return scores
    z = (scores - float(np.max(scores))) / max(1e-6, temperature)
    e = np.exp(z)
    return e / e.sum()


SCORERS = {
    RandomScorer.name: RandomScorer,
    AnomalyStrengthScorer.name: AnomalyStrengthScorer,
    EarliestOnsetScorer.name: EarliestOnsetScorer,
    TransparentScorer.name: TransparentScorer,
    CausalLocalityScorer.name: CausalLocalityScorer,
}
