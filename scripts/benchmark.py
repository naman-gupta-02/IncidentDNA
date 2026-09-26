#!/usr/bin/env python
"""Benchmark the analysis pipeline, stage by stage.

    python scripts/benchmark.py

Answers the question a platform team would actually ask: *can one process keep
up with our telemetry, and where does the time go?*

Each stage is timed in isolation with warm-up and repeats, and reported as a
median of repeats plus tail latencies, because a p99 that blows past the
window interval is what causes a backlog — a good mean hides that.

The headline number is **real-time headroom**: how many services one process
could score inside one 15-second window. Everything is single-process and
single-threaded on purpose; that is the honest baseline to scale from.

Writes `experiments/results/benchmark.json` and `benchmark.md`.
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from incidentdna import topology as topo
from incidentdna.config import ARTIFACTS_DIR, EXPERIMENTS_DIR, SimConfig, WINDOW_SECONDS
from incidentdna.detect.base import observe
from incidentdna.detect.ensemble import EnsembleDetector
from incidentdna.pipeline.analyzer import Analyzer
from incidentdna.pipeline.features import FeatureBuilder, ServiceWindow
from incidentdna.pipeline.store import FeatureStore
from incidentdna.rca.candidates import build_candidates
from incidentdna.rca.ranker import default_ranker
from incidentdna.sim.engine import SimulationEngine


@dataclass
class Timing:
    """Result of timing one operation many times."""

    name: str
    unit: str
    #: Items processed per second (median over repeats).
    throughput: float
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    samples: int
    items_per_call: float

    def to_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def time_it(
    name: str,
    unit: str,
    setup: Callable[[], object],
    call: Callable[[object], int],
    repeats: int = 200,
    warmup: int = 20,
) -> Timing:
    """Time `call`, which returns the number of items it processed.

    `setup` runs before every call and is not timed, so per-call state
    (a fresh store, a fresh window) never leaks between measurements.
    """
    for _ in range(warmup):
        call(setup())

    durations: List[float] = []
    items: List[int] = []
    gc.collect()
    gc.disable()
    try:
        for _ in range(repeats):
            state = setup()
            t0 = time.perf_counter()
            n = call(state)
            durations.append((time.perf_counter() - t0) * 1000.0)
            items.append(n)
    finally:
        gc.enable()

    arr = np.asarray(durations)
    per_call_items = float(np.mean(items))
    median_s = float(np.median(arr)) / 1000.0
    return Timing(
        name=name,
        unit=unit,
        throughput=per_call_items / median_s if median_s > 0 else float("inf"),
        mean_ms=float(arr.mean()),
        p50_ms=float(np.percentile(arr, 50)),
        p95_ms=float(np.percentile(arr, 95)),
        p99_ms=float(np.percentile(arr, 99)),
        max_ms=float(arr.max()),
        samples=len(arr),
        items_per_call=per_call_items,
    )


# --- fixtures --------------------------------------------------------------
def make_window_batches(n_windows: int = 200, seed: int = 5):
    """Real telemetry from the simulator, including an incident."""
    cfg = SimConfig(run_windows=n_windows)
    engine = SimulationEngine(seed=seed, config=cfg, run_id="bench")
    engine.inject("database_latency", severity=0.9, start_window=120, duration_windows=50)
    return [engine.step() for _ in range(n_windows)]


def warm_store(batches, up_to: int = 100):
    """A FeatureStore with enough history for baselines to be available."""
    builder = FeatureBuilder(run_id="bench")
    store = FeatureStore()
    for batch in batches[:up_to]:
        builder.ingest(batch)
        store.extend(builder.pop_ready())
    store.extend(builder.flush())
    return store


def clone_store_with_services(store: FeatureStore, n_services: int) -> FeatureStore:
    """Replicate the real services under synthetic names, to measure how
    detection scales with fleet size. Detection is per-service and does not
    consult the dependency graph, so replication is a faithful stand-in."""
    out = FeatureStore()
    real = topo.SERVICES
    for i in range(n_services):
        src = real[i % len(real)]
        name = src if i < len(real) else f"{src}-replica-{i}"
        for w in store.history(src):
            out.add(
                ServiceWindow(
                    window_index=w.window_index,
                    window_start=w.window_start,
                    service=name,
                    features=dict(w.features),
                    version=w.version,
                    run_id=w.run_id,
                )
            )
    return out


def load_detector() -> tuple:
    path = ARTIFACTS_DIR / "detector.joblib"
    try:
        return EnsembleDetector.load(path), True
    except Exception:
        return EnsembleDetector.statistical_only(), False


# --- the benchmark ---------------------------------------------------------
def machine() -> dict:
    cpu = platform.processor() or platform.machine()
    try:
        if sys.platform == "darwin":
            cpu = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
    except Exception:
        pass
    import os

    return {
        "cpu": cpu,
        "arch": platform.machine(),
        "logical_cores": os.cpu_count(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }


def rss_mb() -> Optional[float]:
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KB, macOS reports bytes.
        return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024
    except Exception:
        return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repeats", type=int, default=200)
    p.add_argument("--out", type=Path, default=EXPERIMENTS_DIR / "results")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    info = machine()
    print(f"machine: {info['cpu']} · {info['logical_cores']} cores · "
          f"Python {info['python']} · single process, single thread\n")

    batches = make_window_batches()
    detector, trained = load_detector()
    print(f"detector: {', '.join(detector.fitted_detectors)}"
          f"{'' if trained else '  (untrained — run make train for the full ensemble)'}\n")

    results: Dict[str, Timing] = {}

    # 1. ingest: raw telemetry records into window buckets
    sample = batches[150]
    results["ingest"] = time_it(
        "Telemetry ingest (bucketing records into windows)",
        "telemetry records/s",
        setup=lambda: FeatureBuilder(run_id="b"),
        call=lambda fb: (fb.ingest(sample), len(sample))[1],
        repeats=args.repeats,
    )

    # 2. reduce: buckets -> feature rows
    def reduce_setup():
        fb = FeatureBuilder(run_id="b")
        fb.ingest(batches[149])
        fb.ingest(batches[150])
        return fb

    results["reduce"] = time_it(
        "Feature reduction (buckets to feature rows)",
        "service-windows/s",
        setup=reduce_setup,
        call=lambda fb: len(fb.pop_ready()),
        repeats=args.repeats,
    )

    # 3. observe: baseline resolution, the hottest path
    store = warm_store(batches, up_to=140)
    probe = [store.history(s)[-1] for s in topo.SERVICES]
    results["observe"] = time_it(
        "Baseline resolution (median/MAD per metric)",
        "service-windows/s",
        setup=lambda: None,
        call=lambda _: [observe(w, store) for w in probe] and len(probe),
        repeats=args.repeats,
    )

    # 4. detection, whole window at a time.
    #    Measured below alongside the statistical-only configuration, so the
    #    two rows come from one measurement and cannot disagree. Ordering
    #    matters here: a stage timed early in the process reads high on cold
    #    caches, which is how this row and the configuration table once
    #    reported 14.4 ms and 8.7 ms for the same operation.

    # 5. which detector configuration should actually ship?
    #    The evaluation says the learned detectors do not improve accuracy;
    #    this measures what they cost, so the trade-off is a number and not
    #    an opinion.
    configs = {
        "statistical_load_adjusted": EnsembleDetector.statistical_only(load_alpha=0.6),
        "full_ensemble": detector,
    }
    config_costs: Dict[str, Timing] = {}
    for cfg_name, cfg in configs.items():
        config_costs[cfg_name] = time_it(
            cfg_name,
            "service-windows/s",
            setup=lambda: None,
            call=lambda _, c=cfg: len(c.score_group(probe, store)),
            repeats=args.repeats,
        )
    cheap = config_costs["statistical_load_adjusted"].p50_ms
    rich = config_costs["full_ensemble"].p50_ms
    config_speedup = rich / cheap if cheap > 0 else float("nan")
    # A copy, so relabelling the stage row does not rename the configuration
    # row — they are the same measurement shown under two headings.
    results["detect"] = dataclasses.replace(
        config_costs["full_ensemble"], name="Anomaly detection (4 detectors, 5 services)"
    )

    # 6. per-detector breakdown
    per_detector: Dict[str, Timing] = {}
    observations = [observe(w, store) for w in probe]
    for name, d in detector.detectors.items():
        per_detector[name] = time_it(
            f"  {name}",
            "service-windows/s",
            setup=lambda: None,
            call=lambda _, d=d: len(d.score_batch(observations)),
            repeats=args.repeats,
        )

    # 7. root-cause ranking + evidence, on a real incident
    analyzer = Analyzer(detector=detector, run_id="bench")
    for batch in batches:
        analyzer.ingest(batch)
    analyzer.finish()
    if not analyzer.incidents:
        print("! no incident produced; RCA benchmark skipped", file=sys.stderr)
        rca = evidence = None
    else:
        incident = analyzer.incidents[0]
        ctx = analyzer.manager.context(incident)
        ranker = default_ranker()
        rca = time_it(
            "Root-cause ranking + evidence (per incident)",
            "incidents/s",
            setup=lambda: None,
            call=lambda _: (ranker.diagnose(ctx), 1)[1],
            repeats=max(30, args.repeats // 4),
        )
        evidence = time_it(
            "  candidate feature extraction only",
            "incidents/s",
            setup=lambda: None,
            call=lambda _: (build_candidates(ctx), 1)[1],
            repeats=max(30, args.repeats // 4),
        )
        results["rca"] = rca
        results["candidates"] = evidence

    # 8. end to end: one window of telemetry through everything
    def e2e_setup():
        a = Analyzer(detector=detector, run_id="b", keep_traces=False)
        for batch in batches[:140]:
            a.ingest(batch)
        return a

    e2e_batch = batches[141]
    results["end_to_end"] = time_it(
        "End to end (telemetry in, diagnosis out, one window)",
        "windows/s",
        setup=e2e_setup,
        call=lambda a: (a.ingest(e2e_batch), 1)[1],
        repeats=40,
        warmup=3,
    )

    # 9. scaling: detection cost against fleet size
    scaling = []
    for n in (5, 10, 25, 50, 100):
        big = clone_store_with_services(store, n)
        probe_n = [big.history(s)[-1] for s in big.services]
        t = time_it(
            f"detection @ {n} services",
            "service-windows/s",
            setup=lambda: None,
            call=lambda _, p=probe_n, s=big: len(detector.score_group(p, s)),
            repeats=60,
            warmup=5,
        )
        scaling.append(
            {
                "services": n,
                "window_ms_p50": round(t.p50_ms, 3),
                "window_ms_p99": round(t.p99_ms, 3),
                "per_service_us": round(t.p50_ms * 1000 / n, 1),
                "throughput": round(t.throughput, 1),
                "window_budget_used_pct": round(t.p99_ms / (WINDOW_SECONDS * 1000) * 100, 4),
            }
        )


    # 10. capacity headroom, from the largest fleet actually measured rather
    #     than from the 5-service point, where per-service overhead dominates.
    #     Derived from p50: the p99 of a 60-repeat scaling measurement is a
    #     single tail sample and swings by 2x between runs, which is far too
    #     noisy to put a headline number on.
    largest = scaling[-1]
    per_service_ms = largest["window_ms_p50"] / largest["services"]
    headroom = int(WINDOW_SECONDS * 1000 / per_service_ms) if per_service_ms > 0 else 0

    report = {
        "machine": info,
        "window_seconds": WINDOW_SECONDS,
        "detector": detector.fitted_detectors,
        "detector_trained": trained,
        "stages": {k: v.to_dict() for k, v in results.items()},
        "per_detector": {k: v.to_dict() for k, v in per_detector.items()},
        "scaling": scaling,
        "detector_configurations": {
            k: v.to_dict() for k, v in config_costs.items()
        },
        "configuration_speedup": round(config_speedup, 2),
        "capacity": {
            "services_per_process_per_window": headroom,
            "basis": (
                f"linear extrapolation from the largest measured fleet "
                f"({largest['services']} services, {largest['window_ms_p50']:.1f} ms p50 "
                f"per window) against one {WINDOW_SECONDS}s window; an "
                f"order-of-magnitude figure, not a measurement"
            ),
        },
        "peak_rss_mb": round(rss_mb() or 0.0, 1),
    }
    (args.out / "benchmark.json").write_text(json.dumps(report, indent=2))
    (args.out / "benchmark.md").write_text(render(report))

    print_report(report)
    print(f"\nwrote {args.out / 'benchmark.json'} and benchmark.md")
    return 0


def print_report(r: dict) -> None:
    print(f"{'stage':52} {'throughput':>22}  {'p50':>8} {'p95':>8} {'p99':>8}")
    print("-" * 104)
    for t in r["stages"].values():
        print(f"{t['name']:52} {t['throughput']:>14,.0f} {t['unit'].split('/')[0][:7]:>7}"
              f"  {t['p50_ms']:>7.3f}ms {t['p95_ms']:>7.3f}ms {t['p99_ms']:>7.3f}ms")
    if r["per_detector"]:
        print("\nper detector:")
        for t in r["per_detector"].values():
            print(f"{t['name']:52} {t['throughput']:>14,.0f} windows"
                  f"  {t['p50_ms']:>7.3f}ms {t['p95_ms']:>7.3f}ms {t['p99_ms']:>7.3f}ms")
    print("\ndetector configuration:")
    for t in r["detector_configurations"].values():
        print(f"{t['name']:52} {t['throughput']:>14,.0f} windows"
              f"  {t['p50_ms']:>7.3f}ms {t['p95_ms']:>7.3f}ms {t['p99_ms']:>7.3f}ms")
    print(f"  -> dropping the learned detectors is {r['configuration_speedup']}x cheaper "
          f"per window")
    print("\nscaling (detection):")
    print(f"  {'services':>9} {'p50/window':>12} {'p99/window':>12} {'per service':>13} {'of 15s budget':>15}")
    for s in r["scaling"]:
        print(f"  {s['services']:>9} {s['window_ms_p50']:>10.2f}ms {s['window_ms_p99']:>10.2f}ms"
              f" {s['per_service_us']:>11.1f}us {s['window_budget_used_pct']:>14.3f}%")
    print(f"\ncapacity: ~{r['capacity']['services_per_process_per_window']:,} services per process "
          f"per {r['window_seconds']}s window (extrapolated, order of magnitude)")
    print(f"peak RSS: {r['peak_rss_mb']} MB")


def render(r: dict) -> str:
    m = r["machine"]
    L = [
        "# IncidentDNA benchmark",
        "",
        f"`{m['cpu']}` · {m['logical_cores']} logical cores · Python {m['python']} · "
        f"NumPy {m['numpy']}",
        "",
        "Single process, single thread. Every stage is timed in isolation with "
        "warm-up and repeats; the throughput column is derived from the median "
        "and the tail columns are what determine whether a backlog forms.",
        "",
        "## Pipeline stages",
        "",
        "| stage | throughput | p50 | p95 | p99 |",
        "|---|---|---|---|---|",
    ]
    for t in r["stages"].values():
        L.append(
            f"| {t['name']} | {t['throughput']:,.0f} {t['unit']} | {t['p50_ms']:.3f} ms "
            f"| {t['p95_ms']:.3f} ms | {t['p99_ms']:.3f} ms |"
        )
    if r["per_detector"]:
        L += ["", "## Cost per detector", "",
              "| detector | throughput | p50 | p99 |", "|---|---|---|---|"]
        for t in r["per_detector"].values():
            L.append(
                f"| `{t['name'].strip()}` | {t['throughput']:,.0f} service-windows/s "
                f"| {t['p50_ms']:.3f} ms | {t['p99_ms']:.3f} ms |"
            )
    L += ["", "## Detector configuration: what the learned models cost", "",
          "| configuration | throughput | p50 | p99 |", "|---|---|---|---|"]
    for t in r["detector_configurations"].values():
        L.append(
            f"| `{t['name']}` | {t['throughput']:,.0f} service-windows/s "
            f"| {t['p50_ms']:.3f} ms | {t['p99_ms']:.3f} ms |"
        )
    L += ["",
          f"Dropping the Isolation Forest and the autoencoder makes detection "
          f"**{r['configuration_speedup']}x cheaper per window** — and the evaluation "
          f"shows it is also the configuration with the better incident precision.",
          "", "## Scaling with fleet size", "",
          "| services | p50 per window | p99 per window | per service | of the 15s budget |",
          "|---|---|---|---|---|"]
    for s in r["scaling"]:
        L.append(
            f"| {s['services']} | {s['window_ms_p50']:.2f} ms | {s['window_ms_p99']:.2f} ms "
            f"| {s['per_service_us']:.1f} µs | {s['window_budget_used_pct']:.3f}% |"
        )
    L += [
        "",
        f"**Capacity:** order of magnitude **{r['capacity']['services_per_process_per_window']:,} "
        f"services** per process per {r['window_seconds']}s window — "
        f"{r['capacity']['basis']}. Memory and Kafka consumer throughput would "
        f"bind long before CPU does.",
        "",
        f"Peak RSS during the benchmark: {r['peak_rss_mb']} MB.",
        "",
    ]
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
