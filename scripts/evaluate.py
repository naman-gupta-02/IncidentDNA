#!/usr/bin/env python
"""Evaluate detection, diagnosis and system performance on the test split.

    python scripts/evaluate.py

Reports, in the three areas the guide asks for:

  detection   incident precision / recall, detection delay, false alerts per
              hour — across four detector configurations so the contribution
              of each idea is visible rather than asserted.
  diagnosis   top-1 / top-3 / MRR for every ranker, including the two naive
              controls and a random floor, on identical incidents.
  system      pipeline throughput and the p95 telemetry-to-diagnosis latency.

Everything comes from runs never seen during training. Writes
`experiments/results/report.json`, `report.md` and PNG charts.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from incidentdna.config import ARTIFACTS_DIR, EXPERIMENTS_DIR, WINDOW_SECONDS
from incidentdna.detect.ensemble import EnsembleDetector
from incidentdna.eval.harness import (
    collect_normal_vectors,
    detection_metrics,
    matched_incidents,
    rank_with,
    replay_all,
)
from incidentdna.eval.report import render_markdown, write_charts
from incidentdna.rca.baseline import (
    AnomalyStrengthScorer,
    CausalLocalityScorer,
    EarliestOnsetScorer,
    RandomScorer,
    TransparentScorer,
)
from incidentdna.rca.learned import LearnedRanker
from incidentdna.rca.ranker import RootCauseRanker
from incidentdna.sim.runs import load_dataset


def build_detectors(vectors: np.ndarray, seed: int) -> Dict[str, EnsembleDetector]:
    """Four configurations, each an ablation of the one before it."""
    configs: Dict[str, EnsembleDetector] = {}
    configs["zscore_only"] = EnsembleDetector(
        detectors=[EnsembleDetector.statistical_only(load_alpha=0.0).detectors["rolling_zscore"]],
        weights={"rolling_zscore": 1.0},
    )
    configs["statistical"] = EnsembleDetector.statistical_only(load_alpha=0.0)
    configs["statistical_load_adjusted"] = EnsembleDetector.statistical_only(load_alpha=0.6)
    full = EnsembleDetector.full(random_state=seed, load_alpha=0.6)
    full.fit(vectors)
    configs["full_ensemble"] = full
    return configs


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=EXPERIMENTS_DIR / "dataset")
    p.add_argument("--artifacts", type=Path, default=ARTIFACTS_DIR)
    p.add_argument("--out", type=Path, default=EXPERIMENTS_DIR / "results")
    p.add_argument("--split", default="test")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-charts", action="store_true")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    frame, metas = load_dataset(args.dataset)
    test_metas = [m for m in metas if m["split"] == args.split]
    print(f"evaluating on {len(test_metas)} {args.split} runs "
          f"({sum(1 for m in test_metas if m['has_incident'])} with an injected fault)")

    report: dict = {
        "split": args.split,
        "runs": len(test_metas),
        "injected_incidents": sum(1 for m in test_metas if m["has_incident"]),
        "window_seconds": WINDOW_SECONDS,
    }

    # --- detection ablation ----------------------------------------------
    print("\n== detection ==")
    vectors = collect_normal_vectors(frame, metas, splits=("train",))
    detectors = build_detectors(vectors, args.seed)
    detection_report: Dict[str, dict] = {}
    replays_by_config = {}
    for name, detector in detectors.items():
        t0 = time.time()
        replays = replay_all(frame, test_metas, detector, splits=(args.split,))
        elapsed = time.time() - t0
        det = detection_metrics(replays)
        replays_by_config[name] = (replays, det)
        detection_report[name] = det.to_dict()
        detection_report[name]["replay_seconds"] = round(elapsed, 2)
        print(
            f"  {name:28} precision={det.precision:.3f} recall={det.recall:.3f} "
            f"f1={det.f1:.3f} median_delay={det.median_detection_delay:.0f}s "
            f"false_alerts/h={det.false_alerts_per_hour:.2f}"
        )
    report["detection"] = detection_report

    # --- diagnosis, on the best detector's incidents ----------------------
    print("\n== diagnosis ==")
    replays, det = replays_by_config["full_ensemble"]
    matches = matched_incidents(replays, det)
    print(f"  ranking {len(matches)} correctly detected incidents")

    rankers: Dict[str, RootCauseRanker] = {
        "random": RootCauseRanker(RandomScorer(seed=args.seed)),
        "naive_anomaly_strength": RootCauseRanker(AnomalyStrengthScorer()),
        "naive_earliest_onset": RootCauseRanker(EarliestOnsetScorer()),
        "transparent": RootCauseRanker(TransparentScorer()),
        "causal_locality": RootCauseRanker(CausalLocalityScorer()),
    }
    for model_type in ("logistic", "gbm"):
        path = args.artifacts / f"ranker_{model_type}.joblib"
        if path.exists():
            rankers[f"learned_{model_type}"] = RootCauseRanker(LearnedRanker.load(path))
        else:
            print(f"  ! {path.name} not found; run scripts/train.py first")

    diagnosis_report: Dict[str, dict] = {}
    for name, ranker in rankers.items():
        res = rank_with(ranker, matches, name=name)
        diagnosis_report[name] = res.to_dict()
        print(
            f"  {name:28} top1={res.top1:.3f} top3={res.top3:.3f} mrr={res.mrr:.3f} "
            f"(coverage={res.candidate_coverage:.3f}, {res.mean_candidates:.1f} candidates)"
        )
    report["diagnosis"] = diagnosis_report

    # --- system performance ----------------------------------------------
    print("\n== system ==")
    perf = measure_performance(frame, test_metas, detectors["full_ensemble"])
    report["system"] = perf
    print(f"  telemetry throughput      {perf['telemetry_events_per_second']:,.0f} events/s")
    print(f"  window throughput         {perf['service_windows_per_second']:,.0f} service-windows/s")
    print(f"  p95 window->diagnosis     {perf['p95_window_to_diagnosis_ms']:.1f} ms")
    print(f"  simulated real-time ratio {perf['realtime_speedup']:,.0f}x")

    (args.out / "report.json").write_text(json.dumps(report, indent=2, default=float))
    md = render_markdown(report)
    (args.out / "report.md").write_text(md)
    print(f"\nwrote {args.out / 'report.json'} and report.md")

    if not args.no_charts:
        try:
            paths = write_charts(report, args.out)
            print("charts: " + ", ".join(p.name for p in paths))
        except Exception as exc:  # pragma: no cover - charts are optional
            print(f"! chart rendering skipped: {exc}")
    return 0


def measure_performance(frame, metas, detector) -> dict:
    """Wall-clock cost of the analysis path, measured on real replays."""
    from incidentdna.pipeline.analyzer import Analyzer
    from incidentdna.sim.runs import frame_to_windows

    sample = [m for m in metas if m["has_incident"]][:8]
    per_window_ms: List[float] = []
    total_windows = 0
    total_rows = 0
    t0 = time.perf_counter()
    for meta in sample:
        windows = frame_to_windows(frame, meta["run_id"])
        total_rows += len(windows)
        analyzer = Analyzer(detector=detector, run_id=meta["run_id"], keep_traces=False)
        analyzer.manager.record_deployments([])
        by_index: Dict[int, list] = {}
        for w in windows:
            by_index.setdefault(w.window_index, []).append(w)
        for widx in sorted(by_index):
            t1 = time.perf_counter()
            analyzer.ingest_windows(by_index[widx])
            per_window_ms.append((time.perf_counter() - t1) * 1000.0)
            total_windows += 1
    elapsed = time.perf_counter() - t0

    # One service-window carries ~46 telemetry records in this deployment
    # (40 duration exemplars + counters + gauges); see sim/engine.py.
    records_per_service_window = int(np.mean([m["record_count"] for m in sample]) /
                                     max(1, total_rows / len(sample)))
    return {
        "runs_measured": len(sample),
        "service_windows_processed": total_rows,
        "wall_clock_seconds": round(elapsed, 3),
        "service_windows_per_second": round(total_rows / elapsed, 1),
        "telemetry_events_per_second": round(total_rows * records_per_service_window / elapsed, 1),
        "mean_window_to_diagnosis_ms": round(float(np.mean(per_window_ms)), 3),
        "p95_window_to_diagnosis_ms": round(float(np.percentile(per_window_ms, 95)), 3),
        "max_window_to_diagnosis_ms": round(float(np.max(per_window_ms)), 3),
        "realtime_speedup": round(total_windows * WINDOW_SECONDS / elapsed, 1),
    }


if __name__ == "__main__":
    raise SystemExit(main())
