"""The end-to-end analysis path, in one object.

    telemetry -> FeatureBuilder -> FeatureStore -> detectors -> IncidentManager

`Analyzer` is used unchanged by the live demo (fed from a bus), by the offline
evaluation harness (fed from stored feature rows) and by the tests. Anything
that only one of those needs lives in the caller, not here.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence

from ..config import WINDOW_SECONDS
from ..detect.base import ServiceAnomaly
from ..detect.ensemble import EnsembleDetector
from ..incidents.manager import IncidentManager
from ..incidents.models import Incident
from ..rca.ranker import RootCauseRanker
from ..telemetry.schema import TelemetryBatch
from .features import FeatureBuilder, ServiceWindow
from .store import FeatureStore


class Analyzer:
    def __init__(
        self,
        detector: Optional[EnsembleDetector] = None,
        ranker: Optional[RootCauseRanker] = None,
        run_id: str = "",
        window_seconds: int = WINDOW_SECONDS,
        keep_traces: bool = True,
    ) -> None:
        self.window_seconds = window_seconds
        self.builder = FeatureBuilder(run_id=run_id, window_seconds=window_seconds)
        self.store = FeatureStore()
        self.detector = detector or EnsembleDetector()
        self.manager = IncidentManager(self.store, ranker=ranker, window_seconds=window_seconds)
        self.keep_traces = keep_traces
        self.windows: List[ServiceWindow] = []
        self.anomalies: List[ServiceAnomaly] = []
        self.processed_windows = 0

    # -- live path ---------------------------------------------------------
    def ingest(self, batch: TelemetryBatch) -> List[ServiceAnomaly]:
        """Feed raw telemetry; returns the anomaly scores for any window that
        became complete as a result."""
        self.builder.ingest(batch)
        if batch.deployments:
            self.manager.record_deployments(batch.deployments)
        if self.keep_traces and batch.spans:
            self.manager.record_traces(batch.traces())
        return self.ingest_windows(self.builder.pop_ready())

    def finish(self) -> List[ServiceAnomaly]:
        """Flush the in-progress window at the end of a run."""
        return self.ingest_windows(self.builder.flush())

    # -- replay path -------------------------------------------------------
    def ingest_windows(self, windows: Sequence[ServiceWindow]) -> List[ServiceAnomaly]:
        if not windows:
            return []
        by_index: Dict[int, List[ServiceWindow]] = defaultdict(list)
        for w in windows:
            by_index[w.window_index].append(w)

        produced: List[ServiceAnomaly] = []
        for widx in sorted(by_index):
            group = by_index[widx]
            # Every service's row for this window must land in the store
            # before scoring, so a service can be compared against its peers.
            # Baselines look strictly *before* `widx`, so this cannot leak.
            for w in group:
                self.store.add(w)
                self.windows.append(w)
            scored = self.detector.score_group(group, self.store)
            self.anomalies.extend(scored)
            produced.extend(scored)
            self.manager.update(widx, scored)
            self.processed_windows += 1
        return produced

    # -- accessors ---------------------------------------------------------
    @property
    def incidents(self) -> List[Incident]:
        return self.manager.incidents

    @property
    def open_incident(self) -> Optional[Incident]:
        return self.manager.open_incident

    def latest_scores(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        widx = self.store.latest_window_index
        for a in reversed(self.anomalies):
            if a.window_index != widx:
                break
            out[a.service] = a.score
        return out
