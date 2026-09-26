#!/usr/bin/env python
"""Fit the detectors and the root-cause rankers.

    python scripts/train.py

Leakage rules enforced here:
  * the unsupervised detectors are fitted only on windows from *normal* runs
    in the *train* split, so "normal" never contains an incident;
  * the learned rankers are fitted only on candidate rows from incidents that
    detection found in the *train* split;
  * the val split is used to pick the operating threshold, the test split is
    never touched.

Artifacts land in `experiments/artifacts/`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from incidentdna.config import ARTIFACTS_DIR, EXPERIMENTS_DIR, SETTINGS
from incidentdna.detect.ensemble import EnsembleDetector
from incidentdna.eval.harness import (
    collect_normal_vectors,
    detection_metrics,
    matched_incidents,
    rank_with,
    replay_all,
    training_rows,
)
from incidentdna.rca.baseline import (
    AnomalyStrengthScorer,
    CausalLocalityScorer,
    EarliestOnsetScorer,
    TransparentScorer,
)
from incidentdna.rca.learned import LearnedRanker, build_training_data
from incidentdna.rca.ranker import RootCauseRanker
from incidentdna.sim.runs import load_dataset


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=EXPERIMENTS_DIR / "dataset")
    p.add_argument("--artifacts", type=Path, default=ARTIFACTS_DIR)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--jobs", type=int, default=1)
    args = p.parse_args()

    t0 = time.time()
    frame, metas = load_dataset(args.dataset)
    print(f"dataset: {len(frame):,} rows / {len(metas)} runs")

    # --- 1. unsupervised detectors on normal train windows ----------------
    print("\n[1/3] fitting anomaly detectors on normal train windows ...")
    vectors = collect_normal_vectors(frame, metas, splits=("train",))
    print(f"      {len(vectors):,} normal detection vectors "
          f"({vectors.shape[1] if len(vectors) else 0} dims)")
    detector = EnsembleDetector.full(random_state=args.seed)
    detector.fit(vectors)
    args.artifacts.mkdir(parents=True, exist_ok=True)
    detector.save(args.artifacts / "detector.joblib")
    print(f"      saved {args.artifacts / 'detector.joblib'}")

    # --- 2. replay train+val to mine candidate rows -----------------------
    print("\n[2/3] replaying train+val runs to mine root-cause candidates ...")
    replays = replay_all(frame, metas, detector, splits=("train", "val"), n_jobs=args.jobs)
    det = detection_metrics(replays)
    print(f"      train+val detection: precision={det.precision:.3f} "
          f"recall={det.recall:.3f} median delay={det.median_detection_delay:.0f}s")
    matches = matched_incidents(replays, det)
    train_matches = [m for m in matches if m[0].split == "train"]
    val_matches = [m for m in matches if m[0].split == "val"]
    rows, groups, truths = training_rows(train_matches)
    print(f"      {len(rows)} candidate rows from {len(train_matches)} train incidents "
          f"({len(val_matches)} val incidents held out)")
    if len(train_matches) < 10:
        print("      ! too few detected training incidents to fit a ranker", file=sys.stderr)
        return 1

    data = build_training_data(rows, groups, truths)
    print(f"      positives: {int(data.labels.sum())} / {len(data)}")

    # --- 3. learned rankers ----------------------------------------------
    print("\n[3/3] fitting root-cause rankers ...")
    report = {
        "dataset_rows": int(len(frame)),
        "runs": len(metas),
        "normal_vectors": int(len(vectors)),
        "train_incidents": len(train_matches),
        "val_incidents": len(val_matches),
        "candidate_rows": len(rows),
        "detection_threshold": SETTINGS.detection.service_score_threshold,
        "rankers": {},
    }
    baselines = {
        "naive_anomaly_strength": RootCauseRanker(AnomalyStrengthScorer()),
        "naive_earliest_onset": RootCauseRanker(EarliestOnsetScorer()),
        "transparent": RootCauseRanker(TransparentScorer()),
        "causal_locality": RootCauseRanker(CausalLocalityScorer()),
    }
    for name, ranker in baselines.items():
        res = rank_with(ranker, val_matches, name=name)
        report["rankers"][name] = {"val_top1": round(res.top1, 4), "val_mrr": round(res.mrr, 4)}
        print(f"      {name:22} val top-1={res.top1:.3f} mrr={res.mrr:.3f} (no training)")

    for model_type in ("logistic", "gbm"):
        ranker = LearnedRanker(model_type=model_type, random_state=args.seed)
        ranker.fit(data)
        path = args.artifacts / f"ranker_{model_type}.joblib"
        ranker.save(path)
        res = rank_with(RootCauseRanker(ranker), val_matches, name=f"learned_{model_type}")
        report["rankers"][f"learned_{model_type}"] = {
            "val_top1": round(res.top1, 4),
            "val_mrr": round(res.mrr, 4),
            "artifact": str(path),
        }
        print(f"      learned_{model_type:13} val top-1={res.top1:.3f} mrr={res.mrr:.3f} -> {path.name}")
        imp = ranker.feature_importance()
        if imp:
            top = sorted(imp.items(), key=lambda kv: -abs(kv[1]))[:6]
            print("        top features: " + ", ".join(f"{k}={v:+.2f}" for k, v in top))
            report["rankers"][f"learned_{model_type}"]["feature_importance"] = {
                k: round(v, 4) for k, v in imp.items()
            }

    report["elapsed_seconds"] = round(time.time() - t0, 1)
    (args.artifacts / "training_report.json").write_text(json.dumps(report, indent=2))
    print(f"\ndone in {report['elapsed_seconds']}s -> {args.artifacts / 'training_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
