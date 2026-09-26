#!/usr/bin/env python
"""Replay a captured telemetry file through the full pipeline.

    python scripts/replay.py capture.jsonl

A capture is produced by running any telemetry producer with
`TELEMETRY_FILE=capture.jsonl` — the real microservices, or the live demo.
Replaying it runs the identical feature, detection and ranking code the live
system uses, which is how the Dockerised services are verified against the
same models the offline evaluation scored.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from incidentdna import topology as topo
from incidentdna.config import ARTIFACTS_DIR, WINDOW_SECONDS
from incidentdna.detect.ensemble import EnsembleDetector
from incidentdna.pipeline.analyzer import Analyzer
from incidentdna.pipeline.features import window_index
from incidentdna.rca.ranker import default_ranker
from incidentdna.telemetry.bus import FileBus
from incidentdna.telemetry.schema import TelemetryBatch


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("capture", type=Path)
    p.add_argument("--detector", type=Path, default=ARTIFACTS_DIR / "detector.joblib")
    p.add_argument("--verbose", action="store_true", help="print every window's scores")
    args = p.parse_args()

    batch = FileBus.read(args.capture)
    print(
        f"{args.capture}: {len(batch.spans):,} spans, {len(batch.metrics):,} metrics, "
        f"{len(batch.logs):,} logs, {len(batch.deployments)} deployments"
    )
    if not len(batch):
        print("nothing to replay")
        return 1

    try:
        detector = EnsembleDetector.load(args.detector)
        print(f"detector: {detector.fitted_detectors}")
    except Exception as exc:
        detector = EnsembleDetector.statistical_only()
        print(f"detector: statistical only ({exc})")

    analyzer = Analyzer(detector=detector, ranker=default_ranker(), run_id="replay")

    # Feed the capture back in window order so the pipeline sees it the way
    # it would have arrived live.
    buckets = defaultdict(TelemetryBatch)
    for s in batch.spans:
        buckets[window_index(s.start_time)].spans.append(s)
    for m in batch.metrics:
        buckets[window_index(m.timestamp)].metrics.append(m)
    for lg in batch.logs:
        buckets[window_index(lg.timestamp)].logs.append(lg)
    for d in batch.deployments:
        buckets[window_index(d.timestamp)].deployments.append(d)

    for widx in sorted(buckets):
        analyzer.ingest(buckets[widx])
        if args.verbose:
            scores = analyzer.latest_scores()
            if scores:
                print(f"  w{widx} " + "  ".join(f"{k}={v:.2f}" for k, v in sorted(scores.items())))
    analyzer.finish()

    print(
        f"\nprocessed {analyzer.processed_windows} windows "
        f"over {len(analyzer.store.services)} services"
    )
    if not analyzer.incidents:
        print("no incidents detected")
        return 0

    for incident in analyzer.incidents:
        print("\n" + "=" * 72)
        print(incident.summary_text(WINDOW_SECONDS))
        if incident.diagnosis:
            print("\nRanking:")
            for c in incident.diagnosis.candidates:
                name = topo.DISPLAY_NAMES.get(c.service, c.service)
                print(f"  {c.rank}. {name:20} score={c.score:.3f} "
                      f"confidence={c.confidence * 100:.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
