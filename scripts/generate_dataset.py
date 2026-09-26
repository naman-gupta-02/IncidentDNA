#!/usr/bin/env python
"""Generate the labelled telemetry dataset.

    python scripts/generate_dataset.py --normal 34 --per-fault 22

Writes `experiments/dataset/windows.parquet` and `runs.json`. Runs are
independent experiments and are the unit of the train/val/test split.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from incidentdna.config import EXPERIMENTS_DIR, SimConfig
from incidentdna.sim import faults as fault_lib
from incidentdna.sim.runs import build_matrix, generate_dataset


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--normal", type=int, default=34, help="runs with no injected fault")
    p.add_argument("--per-fault", type=int, default=22, help="runs per fault type")
    p.add_argument("--seed", type=int, default=20260924)
    p.add_argument("--run-windows", type=int, default=SimConfig().run_windows)
    p.add_argument("--jobs", type=int, default=-1)
    p.add_argument("--out", type=Path, default=EXPERIMENTS_DIR / "dataset")
    p.add_argument(
        "--faults", nargs="*", default=None,
        help=f"subset of {', '.join(fault_lib.FAULT_NAMES)}",
    )
    args = p.parse_args()

    cfg = SimConfig(run_windows=args.run_windows)
    specs = build_matrix(
        normal_runs=args.normal,
        runs_per_fault=args.per_fault,
        seed=args.seed,
        fault_types=args.faults,
    )
    counts = {}
    for s in specs:
        counts[s.fault_type or "normal"] = counts.get(s.fault_type or "normal", 0) + 1
    print(f"experiment matrix: {len(specs)} runs x {cfg.run_windows} windows "
          f"({cfg.run_windows * cfg.window_seconds / 60:.0f} min each)")
    for k, v in sorted(counts.items()):
        print(f"  {k:24} {v:3d} runs")
    splits = {}
    for s in specs:
        splits[s.split] = splits.get(s.split, 0) + 1
    print(f"  split: " + ", ".join(f"{k}={v}" for k, v in sorted(splits.items())))

    generate_dataset(specs, out_dir=args.out, n_jobs=args.jobs, config=cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
