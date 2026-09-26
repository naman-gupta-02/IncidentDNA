"""Render the evaluation report as markdown and charts."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np


def _fmt(v, nd=3):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def render_markdown(report: dict) -> str:
    L: List[str] = []
    L.append("# IncidentDNA evaluation")
    L.append("")
    L.append(
        f"Split: **{report['split']}** — {report['runs']} runs, "
        f"{report['injected_incidents']} injected incidents, "
        f"{report['window_seconds']}s windows. "
        "Runs in this split were never used to fit a detector or a ranker."
    )
    L.append("")

    # detection
    L.append("## Detection")
    L.append("")
    L.append("Alerts are matched to injected incidents as whole events, not rows.")
    L.append("")
    L.append(
        "| detector | precision | recall | F1 | median delay | p95 delay | false alerts/h |"
    )
    L.append("|---|---|---|---|---|---|---|")
    for name, d in report["detection"].items():
        L.append(
            f"| `{name}` | {_fmt(d['incident_precision'])} | {_fmt(d['incident_recall'])} "
            f"| {_fmt(d['f1'])} | {_fmt(d['median_detection_delay_seconds'], 0)} s "
            f"| {_fmt(d['p95_detection_delay_seconds'], 0)} s "
            f"| {_fmt(d['false_alerts_per_hour'], 2)} |"
        )
    L.append("")

    best = report["detection"].get("full_ensemble")
    if best and best.get("by_fault_type"):
        L.append("### Recall by fault type (full ensemble)")
        L.append("")
        L.append("| fault | n | recall | median delay |")
        L.append("|---|---|---|---|")
        for ft, v in sorted(best["by_fault_type"].items()):
            L.append(
                f"| `{ft}` | {v['n']} | {_fmt(v['recall'])} "
                f"| {_fmt(v.get('median_delay_seconds'), 0)} s |"
            )
        L.append("")

    # diagnosis
    L.append("## Diagnosis")
    L.append("")
    first = next(iter(report["diagnosis"].values()), {})
    L.append(
        f"Scored on the {first.get('n_incidents', 0)} incidents the full ensemble detected, "
        f"with a mean of {_fmt(first.get('mean_candidates_per_incident'), 1)} candidate "
        "services per incident. All rankers see identical candidates."
    )
    L.append("")
    L.append("| ranker | top-1 | top-3 | MRR | coverage |")
    L.append("|---|---|---|---|---|")
    for name, d in report["diagnosis"].items():
        L.append(
            f"| `{name}` | {_fmt(d['top1_accuracy'])} | {_fmt(d['top3_accuracy'])} "
            f"| {_fmt(d['mrr'])} | {_fmt(d['candidate_coverage'])} |"
        )
    L.append("")
    L.append(
        "*Coverage* is the fraction of incidents where the true root cause was abnormal "
        "enough to become a candidate at all. It is the ceiling on top-1 for every ranker."
    )
    L.append("")

    ref = report["diagnosis"].get("causal_locality") or first
    if ref.get("by_fault_type"):
        L.append("### Top-1 by fault type (`causal_locality`)")
        L.append("")
        L.append("| fault | n | top-1 | top-3 | MRR |")
        L.append("|---|---|---|---|---|")
        for ft, v in ref["by_fault_type"].items():
            L.append(
                f"| `{ft}` | {v['n']} | {_fmt(v['top1'])} | {_fmt(v['top3'])} | {_fmt(v['mrr'])} |"
            )
        L.append("")
    if ref.get("failures"):
        L.append("### Where it goes wrong")
        L.append("")
        L.append("| run | fault | true cause | predicted | rank of truth | mode |")
        L.append("|---|---|---|---|---|---|")
        for f in ref["failures"][:12]:
            L.append(
                f"| `{f['run_id']}` | `{f['fault_type']}` | `{f['true_root_cause']}` "
                f"| `{f['predicted']}` | {f['rank_of_truth'] or '—'} | {f['mode']} |"
            )
        L.append("")

    # system
    s = report.get("system", {})
    if s:
        L.append("## System performance")
        L.append("")
        L.append("| metric | value |")
        L.append("|---|---|")
        L.append(f"| telemetry events/s | {s['telemetry_events_per_second']:,.0f} |")
        L.append(f"| service-windows/s | {s['service_windows_per_second']:,.0f} |")
        L.append(f"| mean window → diagnosis | {s['mean_window_to_diagnosis_ms']:.1f} ms |")
        L.append(f"| p95 window → diagnosis | {s['p95_window_to_diagnosis_ms']:.1f} ms |")
        L.append(f"| real-time speed-up | {s['realtime_speedup']:,.0f}x |")
        L.append("")
        L.append(
            "Measured single-process on the replay path, which runs the same "
            "feature, detection and ranking code as the live system."
        )
        L.append("")
    return "\n".join(L)


def write_charts(report: dict, out_dir: Path) -> List[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    paths: List[Path] = []
    ink = "#0f172a"
    accent = "#0d9488"
    muted = "#94a3b8"

    # 1. detection ablation
    names = list(report["detection"])
    prec = [report["detection"][n]["incident_precision"] for n in names]
    rec = [report["detection"][n]["incident_recall"] for n in names]
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=160)
    x = np.arange(len(names))
    ax.bar(x - 0.2, prec, 0.38, label="precision", color=accent)
    ax.bar(x + 0.2, rec, 0.38, label="recall", color=muted)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title("Incident detection by detector configuration", color=ink, fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    p = out_dir / "detection_ablation.png"
    fig.savefig(p)
    plt.close(fig)
    paths.append(p)

    # 2. ranker comparison
    rnames = list(report["diagnosis"])
    top1 = [report["diagnosis"][n]["top1_accuracy"] for n in rnames]
    top3 = [report["diagnosis"][n]["top3_accuracy"] for n in rnames]
    mrr = [report["diagnosis"][n]["mrr"] for n in rnames]
    fig, ax = plt.subplots(figsize=(9, 3.8), dpi=160)
    x = np.arange(len(rnames))
    ax.bar(x - 0.26, top1, 0.25, label="top-1", color=accent)
    ax.bar(x, top3, 0.25, label="top-3", color="#5eead4")
    ax.bar(x + 0.26, mrr, 0.25, label="MRR", color=muted)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", "\n") for n in rnames], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title("Root-cause ranking on identical incidents", color=ink, fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    p = out_dir / "ranker_comparison.png"
    fig.savefig(p)
    plt.close(fig)
    paths.append(p)

    # 3. detection delay distribution
    best = report["detection"].get("full_ensemble", {})
    by_ft = best.get("by_fault_type", {})
    if by_ft:
        fts = list(by_ft)
        delays = [by_ft[f].get("median_delay_seconds") or 0 for f in fts]
        fig, ax = plt.subplots(figsize=(7, 3.4), dpi=160)
        ax.barh(fts, delays, color=accent)
        ax.set_xlabel("median detection delay (s)", fontsize=9)
        ax.set_title("How fast each fault type is caught", color=ink, fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        p = out_dir / "detection_delay.png"
        fig.savefig(p)
        plt.close(fig)
        paths.append(p)
    return paths
