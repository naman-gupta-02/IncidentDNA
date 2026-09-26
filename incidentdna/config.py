"""Central configuration for IncidentDNA.

Everything that the simulator, the live demo engine, the offline dataset
builder and the backend need to agree on lives here so that the offline
dataset and the live system are never accidentally computed differently.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = Path(os.environ.get("INCIDENTDNA_EXPERIMENTS", REPO_ROOT / "experiments"))
ARTIFACTS_DIR = EXPERIMENTS_DIR / "artifacts"

# --- Windowing -------------------------------------------------------------
# 15s windows keep anomaly-onset ordering resolvable: typical propagation lags
# in the modelled system are 10-45s, so a downstream service lands 1-3 windows
# after its cause instead of tying with it.
WINDOW_SECONDS = 15

# Rolling baselines used by the change features / z-score detector, in windows.
BASELINE_SHORT_WINDOWS = 20   # ~5 min
BASELINE_LONG_WINDOWS = 240   # ~1 hour (clipped to available history)
MIN_BASELINE_WINDOWS = 8      # below this we do not emit z-scores


@dataclass(frozen=True)
class DetectionConfig:
    """Thresholds for stage one (is something wrong?)."""

    # A service is flagged when its combined anomaly score exceeds this.
    service_score_threshold: float = 0.55
    # ... for this many consecutive windows (debounce against single-window blips).
    consecutive_windows: int = 2
    # An incident stays open while any service is above this (hysteresis).
    clear_threshold: float = 0.35
    # Windows of quiet before an incident is closed.
    close_after_quiet_windows: int = 6
    # Cap on |z| so one wild metric cannot dominate the combined score.
    z_clip: float = 12.0
    # z above which a single metric is considered individually abnormal.
    z_alert: float = 3.5


@dataclass(frozen=True)
class RCAConfig:
    """Weights for the transparent root-cause baseline (guide section 10)."""

    w_anomaly_strength: float = 0.35
    w_early_onset: float = 0.25
    w_propagation_consistency: float = 0.20
    w_deployment_evidence: float = 0.10
    w_log_and_trace_evidence: float = 0.10
    # A deployment is "recent" (full evidence weight) within this many seconds
    # of the candidate's anomaly onset; evidence decays linearly to zero at 2x.
    deployment_recency_seconds: float = 300.0


@dataclass(frozen=True)
class SimConfig:
    """Shape of a single simulated experiment run."""

    window_seconds: int = WINDOW_SECONDS
    run_windows: int = 96                 # 24 minutes per run
    traces_per_window: int = 8            # head-sampled traces kept per window
    fault_start_window_range: tuple = (30, 60)
    fault_duration_windows: tuple = (24, 56)   # 6-14 minutes
    base_rps: float = 40.0


@dataclass(frozen=True)
class Settings:
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    rca: RCAConfig = field(default_factory=RCAConfig)
    sim: SimConfig = field(default_factory=SimConfig)


SETTINGS = Settings()

# --- Feature columns -------------------------------------------------------
# The canonical per-service window feature vector. Order matters: models are
# trained on this order and the backend serves it to the dashboard by name.
FEATURE_COLUMNS = [
    # traffic
    "request_count",
    "concurrent_requests",
    # reliability
    "error_rate",
    "timeout_rate",
    "retry_rate",
    "restart_count",
    # latency
    "latency_p50",
    "latency_p95",
    "latency_p99",
    "self_latency_p95",
    "dep_latency_p95",
    "db_latency_p95",
    # resources
    "cpu_mean",
    "cpu_max",
    "memory_mb",
    "memory_growth_rate",
    "pool_utilization",
    "pool_wait_ms",
    # logs
    "log_error_count",
    "log_warn_count",
    "unique_error_types",
    # traces
    "slow_trace_participation",
]

# Metrics the z-score detector watches. These are the ones where a deviation
# genuinely means "this service is misbehaving" rather than "load changed".
DETECTION_METRICS = [
    "error_rate",
    "timeout_rate",
    "latency_p95",
    "latency_p99",
    "self_latency_p95",
    "db_latency_p95",
    "cpu_mean",
    "memory_growth_rate",
    "pool_utilization",
    "log_error_count",
    "restart_count",
]

# Metrics that are one-sided: only an increase is a problem.
ONE_SIDED_METRICS = set(DETECTION_METRICS)
