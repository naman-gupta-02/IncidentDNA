"""Experiment runs: the labelled-dataset generator.

One *run* is a complete, independent 24-minute experiment: its own traffic
level, its own service versions, its own benign deployments, and at most one
injected fault with known ground truth. Runs are the unit of splitting —
windows from one run never straddle train and test, because windows inside an
incident are strongly correlated and splitting rows at random would leak the
incident's own pattern into the test set.

The matrix follows the guide: ~34 normal runs, ~22 runs per fault type, across
several traffic levels, severities and application versions, with randomised
incident start times.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .. import topology as topo
from ..config import SimConfig, WINDOW_SECONDS
from ..pipeline.features import FeatureBuilder, ServiceWindow
from ..telemetry.schema import DeploymentEvent, TelemetryBatch
from . import faults as fault_lib
from .engine import SimulationEngine

BASE_EPOCH = 1_700_000_000.0
#: Gap between runs so absolute window indices never overlap.
RUN_GAP_WINDOWS = 40

TRAFFIC_LEVELS = [26.0, 40.0, 64.0]
VERSIONS = ["v1.3.0", "v1.4.0", "v1.5.0"]

SPLIT_TRAIN = "train"
SPLIT_VAL = "val"
SPLIT_TEST = "test"


@dataclass
class RunSpec:
    run_id: str
    ordinal: int
    seed: int
    fault_type: Optional[str] = None
    origin: Optional[str] = None
    severity: Optional[float] = None
    base_rps: float = 40.0
    version: str = "v1.4.0"
    split: str = SPLIT_TRAIN
    traffic_spikes: int = 0
    benign_deployments: int = 0

    @property
    def start_time(self) -> float:
        cfg = SimConfig()
        return BASE_EPOCH + self.ordinal * (cfg.run_windows + RUN_GAP_WINDOWS) * cfg.window_seconds


@dataclass
class GroundTruth:
    """The label record from section 7 of the guide."""

    incident_id: str
    run_id: str
    fault_type: str
    root_cause_service: str
    affected_services: List[str]
    start_window: int
    end_window: int
    start_time: float
    end_time: float
    severity: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RunResult:
    spec: RunSpec
    windows: List[ServiceWindow]
    ground_truth: Optional[GroundTruth]
    deployments: List[dict]
    traffic_spikes: List[dict]
    record_count: int

    def rows(self) -> List[dict]:
        return [w.to_row() for w in self.windows]

    def meta(self) -> dict:
        return {
            "run_id": self.spec.run_id,
            "ordinal": self.spec.ordinal,
            "seed": self.spec.seed,
            "split": self.spec.split,
            "base_rps": self.spec.base_rps,
            "version": self.spec.version,
            "fault_type": self.spec.fault_type,
            "has_incident": self.ground_truth is not None,
            "ground_truth": self.ground_truth.to_dict() if self.ground_truth else None,
            "deployments": self.deployments,
            "traffic_spikes": self.traffic_spikes,
            "record_count": self.record_count,
            "window_seconds": WINDOW_SECONDS,
        }


def build_matrix(
    normal_runs: int = 34,
    runs_per_fault: int = 22,
    seed: int = 20260924,
    fault_types: Optional[Sequence[str]] = None,
    train_frac: float = 0.60,
    val_frac: float = 0.15,
) -> List[RunSpec]:
    """The experiment matrix, shuffled so incident types are interleaved."""
    rng = np.random.default_rng(seed)
    fault_types = list(fault_types or fault_lib.FAULT_NAMES)

    plan: List[Tuple[Optional[str], Optional[str]]] = [(None, None)] * normal_runs
    for ft in fault_types:
        spec = fault_lib.get(ft)
        for i in range(runs_per_fault):
            # Cycle deterministically through the fault's possible origins so
            # every origin is represented rather than sampled unevenly.
            origin = spec.origin_candidates[i % len(spec.origin_candidates)]
            plan.append((ft, origin))
    order = rng.permutation(len(plan))

    # Splits are assigned *within* each fault type, by run ordinal. Splitting
    # globally leaves whichever fault types land late in the shuffle with one
    # or two test runs, which makes per-fault recall unreadable. Stratifying
    # keeps every split's mix proportional while the unit of splitting stays
    # the whole run, so no incident's windows straddle a boundary.
    seen: Dict[Optional[str], int] = {}
    totals: Dict[Optional[str], int] = {}
    for ft, _ in plan:
        totals[ft] = totals.get(ft, 0) + 1

    specs: List[RunSpec] = []
    for ordinal, idx in enumerate(order):
        fault_type, origin = plan[int(idx)]
        k = seen.get(fault_type, 0)
        seen[fault_type] = k + 1
        total = totals[fault_type]
        if k < int(total * train_frac):
            split = SPLIT_TRAIN
        elif k < int(total * (train_frac + val_frac)):
            split = SPLIT_VAL
        else:
            split = SPLIT_TEST
        specs.append(
            RunSpec(
                run_id=f"run_{ordinal:04d}",
                ordinal=ordinal,
                seed=int(rng.integers(0, 2**31 - 1)),
                fault_type=fault_type,
                origin=origin,
                severity=float(rng.uniform(0.35, 1.0)) if fault_type else None,
                base_rps=float(rng.choice(TRAFFIC_LEVELS)),
                version=str(rng.choice(VERSIONS)),
                split=split,
                traffic_spikes=int(rng.integers(0, 3)),
                benign_deployments=int(rng.integers(0, 3)),
            )
        )
    return specs


def execute_run(spec: RunSpec, config: Optional[SimConfig] = None) -> RunResult:
    """Simulate one run end to end and reduce it to feature rows + a label."""
    cfg = config or SimConfig()
    rng = np.random.default_rng(spec.seed)
    engine = SimulationEngine(
        seed=spec.seed,
        config=cfg,
        start_time=spec.start_time,
        run_id=spec.run_id,
        base_rps=spec.base_rps,
    )
    engine.versions = {s: spec.version for s in topo.SERVICES}

    # Confounders: benign load surges with no root cause.
    for _ in range(spec.traffic_spikes):
        start = int(rng.integers(8, cfg.run_windows - 20))
        engine.schedule_traffic_spike(
            start_window=start,
            duration_windows=int(rng.integers(8, 22)),
            magnitude=float(rng.uniform(1.6, 2.6)),
        )
    # Confounders: releases unrelated to any fault.
    for _ in range(spec.benign_deployments):
        service = str(rng.choice(topo.SERVICES))
        engine.schedule_deployment(
            window=int(rng.integers(6, cfg.run_windows - 6)),
            service=service,
            version=f"{spec.version}-r{int(rng.integers(1, 9))}",
        )

    fault = None
    if spec.fault_type:
        lo, hi = cfg.fault_start_window_range
        dlo, dhi = cfg.fault_duration_windows
        duration = int(rng.integers(dlo, dhi + 1))
        start = int(rng.integers(lo, min(hi, cfg.run_windows - duration - 4) + 1))
        fault = engine.inject(
            spec.fault_type,
            origin=spec.origin,
            severity=spec.severity,
            duration_windows=duration,
            start_window=start,
        )

    builder = FeatureBuilder(run_id=spec.run_id, window_seconds=cfg.window_seconds)
    records = 0
    for _ in range(cfg.run_windows):
        batch = engine.step()
        records += len(batch)
        builder.ingest(batch)
    windows = builder.flush()
    windows.sort(key=lambda w: (w.window_index, w.service))

    ground_truth = None
    if fault is not None:
        affected = engine.affected_services(fault)
        abs_start = engine.start_window + fault.start_window
        abs_end = engine.start_window + fault.start_window + fault.duration_windows
        ground_truth = GroundTruth(
            incident_id=f"gt_{spec.run_id}",
            run_id=spec.run_id,
            fault_type=fault.spec.name,
            root_cause_service=fault.origin,
            affected_services=affected,
            start_window=abs_start,
            end_window=abs_end,
            start_time=engine.window_start_time(fault.start_window),
            end_time=engine.window_start_time(fault.start_window + fault.duration_windows),
            severity=round(float(fault.severity), 3),
        )

    return RunResult(
        spec=spec,
        windows=windows,
        ground_truth=ground_truth,
        deployments=[d.to_dict() for d in engine.deployments],
        traffic_spikes=[
            {"start_window": s.start_window, "duration_windows": s.duration_windows,
             "magnitude": round(s.magnitude, 3)}
            for s in engine.spikes
        ],
        record_count=records,
    )


def generate_dataset(
    specs: Sequence[RunSpec],
    out_dir: str | Path,
    n_jobs: int = -1,
    config: Optional[SimConfig] = None,
    verbose: bool = True,
) -> Tuple["object", List[dict]]:
    """Run the whole matrix and persist windows.parquet + runs.json."""
    import pandas as pd
    from joblib import Parallel, delayed

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: List[RunResult] = Parallel(n_jobs=n_jobs, verbose=5 if verbose else 0)(
        delayed(execute_run)(spec, config) for spec in specs
    )
    results.sort(key=lambda r: r.spec.ordinal)

    frame = pd.DataFrame([row for r in results for row in r.rows()])
    frame.to_parquet(out_dir / "windows.parquet", index=False)

    metas = [r.meta() for r in results]
    (out_dir / "runs.json").write_text(json.dumps(metas, indent=2))

    if verbose:
        n_inc = sum(1 for m in metas if m["has_incident"])
        print(
            f"wrote {len(frame):,} window rows from {len(metas)} runs "
            f"({n_inc} with an injected fault) to {out_dir}"
        )
    return frame, metas


def load_dataset(out_dir: str | Path):
    """(windows DataFrame, run metadata list)."""
    import pandas as pd

    out_dir = Path(out_dir)
    frame = pd.read_parquet(out_dir / "windows.parquet")
    metas = json.loads((out_dir / "runs.json").read_text())
    return frame, metas


def frame_to_windows(frame, run_id: str) -> List[ServiceWindow]:
    """Rebuild `ServiceWindow` objects for one run so the replay path can
    re-score stored features without re-simulating."""
    from ..config import FEATURE_COLUMNS

    sub = frame[frame["run_id"] == run_id].sort_values(["window_index", "service"])
    out: List[ServiceWindow] = []
    for rec in sub.to_dict("records"):
        out.append(
            ServiceWindow(
                window_index=int(rec["window_index"]),
                window_start=float(rec["window_start"]),
                service=str(rec["service"]),
                features={c: float(rec[c]) for c in FEATURE_COLUMNS},
                version=str(rec["version"]),
                run_id=run_id,
                deployed_in_window=bool(rec["deployed_in_window"]),
            )
        )
    return out
