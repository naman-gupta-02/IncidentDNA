"""Replay harness: stored feature rows -> alerts, candidates, metrics.

Detection runs once per run and is ranker-independent, so every ranker is
scored on exactly the same incidents and the same candidate rows. That is the
only way a "learned model beats the baseline" claim means anything.

Leakage control:
  * splits are by whole run, never by row;
  * the unsupervised detectors see only windows from *normal train runs*;
  * the learned rankers see only candidate rows from *train* incidents;
  * every reported number comes from the test split.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import WINDOW_SECONDS
from ..detect.base import detection_vector
from ..detect.ensemble import EnsembleDetector
from ..pipeline.analyzer import Analyzer
from ..pipeline.store import FeatureStore
from ..rca.candidates import IncidentContext, build_candidates
from ..rca.ranker import RootCauseRanker
from ..sim.runs import frame_to_windows
from ..telemetry.schema import DeploymentEvent
from .detection import AlertRecord, DetectionResult, TruthRecord, evaluate_detection
from .diagnosis import DiagnosisResult, RankedIncident


@dataclass
class ReplayedRun:
    run_id: str
    split: str
    fault_type: Optional[str]
    truth: Optional[TruthRecord]
    alerts: List[AlertRecord] = field(default_factory=list)
    n_windows: int = 0
    #: incident_id -> candidate feature rows (ranker input)
    candidates: Dict[str, List[Dict[str, float]]] = field(default_factory=dict)
    #: incident_id -> detected window, for matching against ground truth
    incident_windows: Dict[str, Tuple[int, int]] = field(default_factory=dict)


def _deployments(meta: dict) -> List[DeploymentEvent]:
    return [DeploymentEvent(**d) for d in meta.get("deployments", [])]


def _truth(meta: dict) -> Optional[TruthRecord]:
    gt = meta.get("ground_truth")
    if not gt:
        return None
    return TruthRecord(
        run_id=meta["run_id"],
        fault_type=gt["fault_type"],
        root_cause_service=gt["root_cause_service"],
        start_window=int(gt["start_window"]),
        end_window=int(gt["end_window"]),
    )


def replay_run(
    frame,
    meta: dict,
    detector: EnsembleDetector,
    window_seconds: int = WINDOW_SECONDS,
) -> ReplayedRun:
    """Run one stored run through detection and extract candidate rows."""
    run_id = meta["run_id"]
    windows = frame_to_windows(frame, run_id)
    analyzer = Analyzer(
        detector=detector, run_id=run_id, window_seconds=window_seconds, keep_traces=False
    )
    analyzer.manager.record_deployments(_deployments(meta))
    analyzer.ingest_windows(windows)

    out = ReplayedRun(
        run_id=run_id,
        split=meta["split"],
        fault_type=meta.get("fault_type"),
        truth=_truth(meta),
        n_windows=len({w.window_index for w in windows}),
    )
    for incident in analyzer.incidents:
        out.alerts.append(
            AlertRecord(
                run_id=run_id,
                incident_id=incident.incident_id,
                detected_window=incident.detected_window,
                start_window=incident.start_window,
                end_window=incident.end_window,
                severity=incident.severity,
            )
        )
        ctx = analyzer.manager.context(incident)
        out.candidates[incident.incident_id] = build_candidates(ctx)
        out.incident_windows[incident.incident_id] = (
            incident.start_window,
            incident.end_window or analyzer.store.latest_window_index,
        )
    return out


def collect_normal_vectors(
    frame, metas: Sequence[dict], splits: Sequence[str] = ("train",)
) -> np.ndarray:
    """Detection vectors from runs with no injected fault, for unsupervised fits."""
    vectors: List[np.ndarray] = []
    for meta in metas:
        if meta["split"] not in splits or meta["has_incident"]:
            continue
        store = FeatureStore()
        for w in frame_to_windows(frame, meta["run_id"]):
            store.add(w)
            vec, _ = detection_vector(w, store)
            if vec is not None:
                vectors.append(vec)
    return np.vstack(vectors) if vectors else np.empty((0, 0))


def replay_all(
    frame,
    metas: Sequence[dict],
    detector: EnsembleDetector,
    splits: Optional[Sequence[str]] = None,
    window_seconds: int = WINDOW_SECONDS,
    n_jobs: int = 1,
) -> List[ReplayedRun]:
    targets = [m for m in metas if splits is None or m["split"] in splits]
    if n_jobs == 1:
        return [replay_run(frame, m, detector, window_seconds) for m in targets]
    from joblib import Parallel, delayed

    return list(
        Parallel(n_jobs=n_jobs)(
            delayed(replay_run)(frame, m, detector, window_seconds) for m in targets
        )
    )


def detection_metrics(
    replays: Sequence[ReplayedRun], grace_windows: int = 8, window_seconds: int = WINDOW_SECONDS
) -> DetectionResult:
    return evaluate_detection(
        alerts_by_run={r.run_id: r.alerts for r in replays},
        truth_by_run={r.run_id: r.truth for r in replays},
        normal_windows_by_run={r.run_id: r.n_windows for r in replays},
        grace_windows=grace_windows,
        window_seconds=window_seconds,
    )


def matched_incidents(
    replays: Sequence[ReplayedRun], detection: DetectionResult
) -> List[Tuple[ReplayedRun, str, TruthRecord]]:
    """(run, incident_id, truth) for every correctly detected incident."""
    by_run = {r.run_id: r for r in replays}
    out = []
    for m in detection.matched:
        run = by_run[m["run_id"]]
        if run.truth is not None:
            out.append((run, m["incident_id"], run.truth))
    return out


def rank_with(
    ranker: RootCauseRanker,
    matches: Sequence[Tuple[ReplayedRun, str, TruthRecord]],
    name: Optional[str] = None,
) -> DiagnosisResult:
    result = DiagnosisResult(ranker=name or ranker.name)
    for run, incident_id, truth in matches:
        rows = run.candidates.get(incident_id, [])
        if not rows:
            continue
        scores = ranker.rank_rows(rows)
        order = np.argsort(-scores)
        ranking = [rows[int(i)]["_service"] for i in order]
        result.incidents.append(
            RankedIncident(
                run_id=run.run_id,
                incident_id=incident_id,
                fault_type=truth.fault_type,
                true_root_cause=truth.root_cause_service,
                ranking=ranking,
                scores=[float(scores[int(i)]) for i in order],
            )
        )
    return result


def training_rows(
    matches: Sequence[Tuple[ReplayedRun, str, TruthRecord]]
) -> Tuple[List[Dict[str, float]], List[str], List[str]]:
    """(candidate rows, incident ids, true root cause per row)."""
    rows: List[Dict[str, float]] = []
    groups: List[str] = []
    truths: List[str] = []
    for run, incident_id, truth in matches:
        for row in run.candidates.get(incident_id, []):
            rows.append(row)
            groups.append(f"{run.run_id}:{incident_id}")
            truths.append(truth.root_cause_service)
    return rows, groups, truths
