"""The live system behind the dashboard.

A background thread advances a telemetry source one window at a time, publishes
the records to the bus, and feeds whatever the bus yields into the `Analyzer`.
Everything downstream of `bus.publish` is the same code the offline evaluation
runs, so what the dashboard shows is not a separate demo implementation.

Two sources are supported:

``simulated``
    The generative engine from `incidentdna.sim`, advanced faster than real
    time so a recruiter watching the page sees an incident develop in seconds
    rather than minutes. Fault injection is a method call.

``kafka``
    Records produced by the real Dockerised microservices, consumed from the
    Kafka topics. Fault injection is an HTTP call to the service's own fault
    endpoint. Selected with `INCIDENTDNA_SOURCE=kafka`.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from incidentdna import topology as topo
from incidentdna.config import ARTIFACTS_DIR, SETTINGS, SimConfig, WINDOW_SECONDS
from incidentdna.detect.ensemble import EnsembleDetector
from incidentdna.pipeline.analyzer import Analyzer
from incidentdna.rca.baseline import SCORERS, CausalLocalityScorer
from incidentdna.rca.learned import LearnedRanker
from incidentdna.rca.ranker import RootCauseRanker
from incidentdna.sim import faults as fault_lib
from incidentdna.sim.engine import SimulationEngine
from incidentdna.telemetry.bus import InMemoryBus, TelemetryBus, TeeBus
from incidentdna.telemetry.schema import TelemetryBatch

log = logging.getLogger("incidentdna.live")

#: Windows simulated before the clock starts, so every service has a baseline
#: and the first injected fault is detectable immediately.
WARMUP_WINDOWS = 40


class TelemetrySource:
    """Produces one window of telemetry per call."""

    def step(self) -> TelemetryBatch:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> None:
        pass


class SimulatedSource(TelemetrySource):
    def __init__(self, seed: int = 7, base_rps: float = 40.0) -> None:
        self.config = SimConfig(run_windows=10**9)
        self.engine = SimulationEngine(
            seed=seed,
            config=self.config,
            start_time=time.time(),
            run_id="live",
            base_rps=base_rps,
        )

    def step(self) -> TelemetryBatch:
        return self.engine.step()


class KafkaSource(TelemetrySource):
    """Drains whatever the real services published since the last window."""

    def __init__(self, bootstrap_servers: str) -> None:
        from incidentdna.telemetry.bus import KafkaBus

        self.bus = KafkaBus(bootstrap_servers=bootstrap_servers, produce=False, consume=True)

    def step(self) -> TelemetryBatch:
        return self.bus.drain()

    def close(self) -> None:
        self.bus.close()


@dataclass
class InjectedFault:
    fault_id: str
    fault_type: str
    origin: str
    severity: float
    started_at: float
    duration_seconds: float

    def to_dict(self) -> dict:
        remaining = max(0.0, self.duration_seconds - (time.time() - self.started_at))
        return {
            "fault_id": self.fault_id,
            "fault_type": self.fault_type,
            "origin": self.origin,
            "origin_display": topo.DISPLAY_NAMES.get(self.origin, self.origin),
            "severity": round(self.severity, 2),
            "duration_seconds": self.duration_seconds,
            "remaining_seconds": round(remaining, 1),
            "active": remaining > 0,
        }


class LiveSystem:
    """Owns the source thread, the analyzer and the injected-fault log."""

    def __init__(
        self,
        source_kind: str = "simulated",
        speed: float = 10.0,
        seed: int = 7,
        detector_path: Optional[str] = None,
        ranker_name: str = "causal_locality",
        capture_path: Optional[str] = None,
    ) -> None:
        self.speed = max(0.1, speed)
        self.source_kind = source_kind
        self.lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self.bus: TelemetryBus = InMemoryBus()
        if capture_path:
            from incidentdna.telemetry.bus import FileBus

            self.bus = TeeBus(self.bus, FileBus(capture_path, mode="w"))

        if source_kind == "kafka":
            self.source: TelemetrySource = KafkaSource(
                os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
            )
        else:
            self.source = SimulatedSource(seed=seed)

        self.detector = self._load_detector(detector_path)
        self.rankers = self._load_rankers()
        self.ranker_name = ranker_name if ranker_name in self.rankers else "causal_locality"
        self.analyzer = Analyzer(
            detector=self.detector,
            ranker=self.rankers[self.ranker_name],
            run_id="live",
            window_seconds=WINDOW_SECONDS,
        )
        self.injected: List[InjectedFault] = []
        self.started_at = time.time()
        self.warmed_up = False
        self.telemetry_records = 0

    # -- setup -------------------------------------------------------------
    @staticmethod
    def _load_detector(path: Optional[str]) -> EnsembleDetector:
        candidate = path or str(ARTIFACTS_DIR / "detector.joblib")
        try:
            detector = EnsembleDetector.load(candidate)
            log.info("loaded trained detector from %s (%s)", candidate, detector.fitted_detectors)
            return detector
        except Exception as exc:
            log.warning(
                "no trained detector at %s (%s); falling back to the statistical "
                "ensemble, which needs no training",
                candidate,
                exc,
            )
            return EnsembleDetector.statistical_only()

    @staticmethod
    def _load_rankers() -> Dict[str, RootCauseRanker]:
        rankers: Dict[str, RootCauseRanker] = {
            name: RootCauseRanker(cls()) for name, cls in SCORERS.items()
        }
        for model_type in ("logistic", "gbm"):
            path = ARTIFACTS_DIR / f"ranker_{model_type}.joblib"
            if path.exists():
                try:
                    rankers[f"learned_{model_type}"] = RootCauseRanker(LearnedRanker.load(path))
                except Exception as exc:  # pragma: no cover - stale artifact
                    log.warning("skipping %s: %s", path.name, exc)
        return rankers

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            return
        self._warm_up()
        self._thread = threading.Thread(target=self._run, name="incidentdna-live", daemon=True)
        self._thread.start()
        log.info("live system running (%s source, %.1fx speed)", self.source_kind, self.speed)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        self.source.close()
        self.bus.close()

    def _warm_up(self) -> None:
        """Build baselines before the clock starts. Without history the
        detectors abstain, so a fault injected in the first minute would be
        invisible rather than merely undetected."""
        if self.source_kind != "simulated":
            return
        with self.lock:
            for _ in range(WARMUP_WINDOWS):
                self._tick()
            self.warmed_up = True
        log.info("warm-up complete: %d windows of baseline", WARMUP_WINDOWS)

    def _tick(self) -> None:
        batch = self.source.step()
        self.bus.publish(batch)
        drained = self.bus.drain()
        self.telemetry_records += len(drained)
        self.analyzer.ingest(drained)

    def _run(self) -> None:
        interval = WINDOW_SECONDS / self.speed
        while not self._stop.is_set():
            started = time.perf_counter()
            try:
                with self.lock:
                    self._tick()
                    self._expire_faults()
            except Exception:  # pragma: no cover - keep the demo alive
                log.exception("live tick failed")
            self._stop.wait(max(0.0, interval - (time.perf_counter() - started)))

    def _expire_faults(self) -> None:
        now = time.time()
        for f in self.injected:
            if now - f.started_at > f.duration_seconds:
                f.duration_seconds = min(f.duration_seconds, now - f.started_at)

    # -- control -----------------------------------------------------------
    def inject(
        self,
        fault_type: str,
        origin: Optional[str] = None,
        severity: float = 0.85,
        duration_seconds: float = 180.0,
    ) -> InjectedFault:
        if self.source_kind != "simulated":
            raise RuntimeError(
                "fault injection over the Kafka source is done by calling the "
                "target service's /fault endpoint directly (see services/common/faults.py)"
            )
        with self.lock:
            engine = self.source.engine  # type: ignore[attr-defined]
            windows = max(4, int(duration_seconds / WINDOW_SECONDS))
            active = engine.inject(
                fault_type, origin=origin, severity=severity, duration_windows=windows
            )
            record = InjectedFault(
                fault_id=active.fault_id,
                fault_type=fault_type,
                origin=active.origin,
                severity=active.severity,
                started_at=time.time(),
                # In wall-clock terms the fault lasts `speed` times less long.
                duration_seconds=windows * WINDOW_SECONDS / self.speed,
            )
            self.injected.append(record)
            log.info("injected %s at %s (severity %.2f)", fault_type, active.origin, active.severity)
            return record

    def clear_faults(self) -> int:
        with self.lock:
            engine = getattr(self.source, "engine", None)
            if engine is None:
                return 0
            n = len(engine.active_faults)
            engine.completed_faults.extend(engine.active_faults)
            engine.active_faults.clear()
            for f in self.injected:
                f.duration_seconds = min(f.duration_seconds, time.time() - f.started_at)
            return n

    def set_ranker(self, name: str) -> str:
        with self.lock:
            if name not in self.rankers:
                raise KeyError(name)
            self.ranker_name = name
            self.analyzer.manager.ranker = self.rankers[name]
            self.analyzer.manager.rediagnose_all(self.rankers[name])
            return name

    # -- read models -------------------------------------------------------
    def service_states(self) -> List[dict]:
        with self.lock:
            scores = self.analyzer.latest_scores()
            open_incident = self.analyzer.open_incident
            root_cause = (
                open_incident.diagnosis.root_cause
                if open_incident and open_incident.diagnosis
                else None
            )
            out = []
            for service in topo.SERVICES:
                window = self.analyzer.store.latest(service)
                score = scores.get(service, 0.0)
                out.append(
                    {
                        "service": service,
                        "display_name": topo.DISPLAY_NAMES[service],
                        "score": round(score, 4),
                        "health": _health(score),
                        "is_root_cause": service == root_cause,
                        "in_incident": bool(open_incident and service in open_incident.services),
                        "metrics": {
                            k: round(window.get(k), 4)
                            for k in (
                                "request_count",
                                "error_rate",
                                "latency_p95",
                                "self_latency_p95",
                                "db_latency_p95",
                                "cpu_mean",
                                "memory_mb",
                                "pool_utilization",
                                "timeout_rate",
                            )
                        }
                        if window
                        else {},
                    }
                )
            return out

    def snapshot(self) -> dict:
        with self.lock:
            open_incident = self.analyzer.open_incident
            return {
                "source": self.source_kind,
                "speed": self.speed,
                "window_seconds": WINDOW_SECONDS,
                "warmed_up": self.warmed_up,
                "uptime_seconds": round(time.time() - self.started_at, 1),
                "windows_processed": self.analyzer.processed_windows,
                "telemetry_records": self.telemetry_records,
                "latest_window": self.analyzer.store.latest_window_index,
                "simulated_clock": (
                    self.analyzer.store.latest_window_index * WINDOW_SECONDS
                    if self.analyzer.store.latest_window_index > 0
                    else 0
                ),
                "detector": self.detector.fitted_detectors,
                "ranker": self.ranker_name,
                "available_rankers": sorted(self.rankers),
                "services": self.service_states(),
                "open_incident_id": open_incident.incident_id if open_incident else None,
                "incident_count": len(self.analyzer.incidents),
                "active_faults": [f.to_dict() for f in self.injected if f.to_dict()["active"]],
            }

    def metric_series(self, services: List[str], metrics: List[str], limit: int = 160) -> dict:
        with self.lock:
            out: Dict[str, Dict[str, list]] = {}
            for service in services:
                history = self.analyzer.store.history(service)[-limit:]
                out[service] = {
                    "window_index": [w.window_index for w in history],
                    "time": [w.window_start for w in history],
                    **{m: [round(w.get(m), 4) for w in history] for m in metrics},
                }
            scores: Dict[str, list] = {s: [] for s in services}
            windows: List[int] = []
            for entry in list(self.analyzer.manager.recent_scores)[-limit:]:
                windows.append(entry["window_index"])
                for s in services:
                    scores[s].append(entry["scores"].get(s, 0.0))
            return {"metrics": out, "scores": {"window_index": windows, "series": scores}}

    def traces(self, incident_id: Optional[str] = None, limit: int = 8) -> List[dict]:
        """Representative slow traces, newest first."""
        with self.lock:
            candidates = list(self.analyzer.builder.recent_traces)
            incident = self.analyzer.manager.by_id(incident_id) if incident_id else None
            if incident is not None:
                lo = incident.start_time - 30
                hi = (incident.end_time or float("inf")) + 30
                candidates = [t for t in candidates if lo <= t.start_time <= hi]
            candidates.sort(key=lambda t: -t.duration_ms)
            return [t.to_dict() for t in candidates[:limit]]

    def incident_payload(self, incident_id: str) -> Optional[dict]:
        with self.lock:
            incident = self.analyzer.manager.by_id(incident_id)
            if incident is None:
                return None
            payload = incident.to_dict(WINDOW_SECONDS)
            payload["summary_text"] = incident.summary_text(WINDOW_SECONDS)
            payload["traces"] = self.traces(incident_id, limit=5)
            return payload

    def incidents(self) -> List[dict]:
        with self.lock:
            return self.analyzer.manager.as_list()


def _health(score: float) -> str:
    if score >= SETTINGS.detection.service_score_threshold:
        return "critical"
    if score >= SETTINGS.detection.clear_threshold:
        return "degraded"
    return "healthy"
