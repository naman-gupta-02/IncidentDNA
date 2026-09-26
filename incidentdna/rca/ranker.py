"""Stage two orchestration: incident in, ranked evidence-backed diagnosis out."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from ..incidents.models import Candidate, Diagnosis
from .baseline import CausalLocalityScorer, TransparentScorer, WeightedScorer, softmax_confidence
from .candidates import IncidentContext, build_candidates
from .evidence import build_evidence
from .learned import LearnedRanker

Scorer = Union[WeightedScorer, LearnedRanker]


class RootCauseRanker:
    """Ranks candidates with whichever scorer is configured.

    The scorer is swappable precisely so the dashboard and the evaluation
    harness can put the transparent baseline and a learned model side by side
    on the same incident.
    """

    def __init__(self, scorer: Optional[Scorer] = None) -> None:
        self.scorer: Scorer = scorer or CausalLocalityScorer()

    @property
    def name(self) -> str:
        return getattr(self.scorer, "name", "unknown")

    def rank_rows(self, rows: Sequence[Dict[str, float]]) -> np.ndarray:
        if not rows:
            return np.array([])
        return np.asarray(self.scorer.score(list(rows)), dtype=float)

    def diagnose(self, ctx: IncidentContext) -> Optional[Diagnosis]:
        rows = build_candidates(ctx)
        if not rows:
            return None
        scores = self.rank_rows(rows)
        confidences = softmax_confidence(scores)
        order = np.argsort(-scores)

        candidates: List[Candidate] = []
        for rank, idx in enumerate(order, start=1):
            row = rows[int(idx)]
            service = row["_service"]
            state = ctx.states[service]
            support, against = build_evidence(ctx, row, state)
            candidates.append(
                Candidate(
                    service=service,
                    features={k: v for k, v in row.items() if not k.startswith("_")},
                    score=float(scores[int(idx)]),
                    rank=rank,
                    confidence=float(confidences[int(idx)]),
                    evidence=support,
                    contradicting=against,
                    contributions=self._contributions(row),
                )
            )
        return Diagnosis(
            ranker=self.name,
            candidates=candidates,
            generated_at_window=ctx.store.latest_window_index,
        )

    def _contributions(self, row: Dict[str, float]) -> Dict[str, float]:
        fn = getattr(self.scorer, "contributions", None)
        return fn(row) if callable(fn) else {}


def default_ranker() -> RootCauseRanker:
    return RootCauseRanker(CausalLocalityScorer())


def transparent_ranker() -> RootCauseRanker:
    return RootCauseRanker(TransparentScorer())
