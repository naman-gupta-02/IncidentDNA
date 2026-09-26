"""Stage-two evaluation: top-1, top-3 and MRR over correctly detected incidents.

Diagnosis is only scored on incidents that detection actually caught — ranking
quality on an incident nobody was told about is not a meaningful number. The
two failure modes are reported separately:

* **miss**      - the true root cause was never a candidate (it never went
                  abnormal, so no ranker could have found it). This bounds
                  every ranker's achievable accuracy.
* **misrank**   - the true cause was a candidate but placed below another.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass
class RankedIncident:
    run_id: str
    incident_id: str
    fault_type: str
    true_root_cause: str
    ranking: List[str]
    scores: List[float] = field(default_factory=list)
    confidence: float = 0.0

    @property
    def in_candidates(self) -> bool:
        return self.true_root_cause in self.ranking

    @property
    def rank(self) -> Optional[int]:
        if not self.in_candidates:
            return None
        return self.ranking.index(self.true_root_cause) + 1

    @property
    def reciprocal_rank(self) -> float:
        r = self.rank
        return 1.0 / r if r else 0.0


@dataclass
class DiagnosisResult:
    ranker: str
    incidents: List[RankedIncident] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.incidents)

    def _frac(self, fn) -> float:
        return float(np.mean([fn(i) for i in self.incidents])) if self.incidents else 0.0

    @property
    def top1(self) -> float:
        return self._frac(lambda i: 1.0 if i.rank == 1 else 0.0)

    @property
    def top3(self) -> float:
        return self._frac(lambda i: 1.0 if (i.rank is not None and i.rank <= 3) else 0.0)

    @property
    def mrr(self) -> float:
        return self._frac(lambda i: i.reciprocal_rank)

    @property
    def candidate_coverage(self) -> float:
        """Upper bound on top-1: how often the true cause was even ranked."""
        return self._frac(lambda i: 1.0 if i.in_candidates else 0.0)

    @property
    def mean_candidates(self) -> float:
        return float(np.mean([len(i.ranking) for i in self.incidents])) if self.incidents else 0.0

    def by_fault_type(self) -> Dict[str, Dict[str, float]]:
        groups: Dict[str, List[RankedIncident]] = {}
        for i in self.incidents:
            groups.setdefault(i.fault_type, []).append(i)
        return {
            ft: {
                "n": len(items),
                "top1": float(np.mean([1.0 if i.rank == 1 else 0.0 for i in items])),
                "top3": float(
                    np.mean([1.0 if (i.rank is not None and i.rank <= 3) else 0.0 for i in items])
                ),
                "mrr": float(np.mean([i.reciprocal_rank for i in items])),
            }
            for ft, items in sorted(groups.items())
        }

    def confusion(self) -> Dict[str, Dict[str, int]]:
        """true root cause -> predicted root cause -> count."""
        out: Dict[str, Dict[str, int]] = {}
        for i in self.incidents:
            pred = i.ranking[0] if i.ranking else "none"
            out.setdefault(i.true_root_cause, {}).setdefault(pred, 0)
            out[i.true_root_cause][pred] += 1
        return out

    def failures(self, limit: int = 20) -> List[dict]:
        out = []
        for i in self.incidents:
            if i.rank == 1:
                continue
            out.append(
                {
                    "run_id": i.run_id,
                    "fault_type": i.fault_type,
                    "true_root_cause": i.true_root_cause,
                    "predicted": i.ranking[0] if i.ranking else None,
                    "rank_of_truth": i.rank,
                    "mode": "miss" if i.rank is None else "misrank",
                    "ranking": i.ranking,
                }
            )
        return out[:limit]

    def to_dict(self) -> dict:
        return {
            "ranker": self.ranker,
            "n_incidents": self.n,
            "top1_accuracy": round(self.top1, 4),
            "top3_accuracy": round(self.top3, 4),
            "mrr": round(self.mrr, 4),
            "candidate_coverage": round(self.candidate_coverage, 4),
            "mean_candidates_per_incident": round(self.mean_candidates, 2),
            "by_fault_type": self.by_fault_type(),
            "confusion": self.confusion(),
            "failures": self.failures(),
        }
