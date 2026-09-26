"""In-memory feature store with rolling baselines.

Holds the recent history of every service's window features and answers the
two questions the detectors ask: "what does normal look like for this service
right now?" and "what did this service look like N windows ago?".

Baseline lookups are the hottest path in the whole system — every detector
asks for one per metric per service per window — so history is kept as a
sorted list with a parallel index array and sliced by bisection, making a
lookup O(lookback) rather than O(history).
"""
from __future__ import annotations

import bisect
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from ..config import (
    BASELINE_LONG_WINDOWS,
    BASELINE_SHORT_WINDOWS,
    MIN_BASELINE_WINDOWS,
)
from .features import ServiceWindow


class FeatureStore:
    def __init__(self, max_windows: int = BASELINE_LONG_WINDOWS * 2) -> None:
        self.max_windows = max_windows
        self._by_service: Dict[str, List[ServiceWindow]] = defaultdict(list)
        self._index: Dict[str, List[int]] = defaultdict(list)
        self.latest_window_index: int = -1

    # -- writing -----------------------------------------------------------
    def add(self, window: ServiceWindow) -> None:
        hist = self._by_service[window.service]
        idx = self._index[window.service]
        if idx and window.window_index < idx[-1]:
            # Out-of-order arrival: insert in place so bisection stays valid.
            pos = bisect.bisect_left(idx, window.window_index)
            hist.insert(pos, window)
            idx.insert(pos, window.window_index)
        else:
            hist.append(window)
            idx.append(window.window_index)
        if len(hist) > self.max_windows:
            drop = len(hist) - self.max_windows
            del hist[:drop]
            del idx[:drop]
        self.latest_window_index = max(self.latest_window_index, window.window_index)

    def extend(self, windows: Iterable[ServiceWindow]) -> None:
        for w in windows:
            self.add(w)

    # -- reading -----------------------------------------------------------
    @property
    def services(self) -> List[str]:
        return sorted(self._by_service)

    def history(self, service: str) -> List[ServiceWindow]:
        return list(self._by_service.get(service, ()))

    def latest(self, service: str) -> Optional[ServiceWindow]:
        h = self._by_service.get(service)
        return h[-1] if h else None

    def at(self, service: str, window_index: int) -> Optional[ServiceWindow]:
        idx = self._index.get(service)
        if not idx:
            return None
        pos = bisect.bisect_left(idx, window_index)
        if pos < len(idx) and idx[pos] == window_index:
            return self._by_service[service][pos]
        return None

    def series(self, service: str, metric: str) -> Tuple[List[int], List[float]]:
        h = self._by_service.get(service, ())
        return [w.window_index for w in h], [w.get(metric) for w in h]

    def window_slice(
        self, service: str, lo_window: int, hi_window: int
    ) -> List[ServiceWindow]:
        """Windows with `lo_window <= index <= hi_window`."""
        idx = self._index.get(service)
        if not idx:
            return []
        lo = bisect.bisect_left(idx, lo_window)
        hi = bisect.bisect_right(idx, hi_window)
        return self._by_service[service][lo:hi]

    # -- baselines ---------------------------------------------------------
    def _lookback_values(
        self, service: str, metric: str, before_window: int, lookback: int
    ) -> Optional[np.ndarray]:
        idx = self._index.get(service)
        if not idx:
            return None
        hi = bisect.bisect_left(idx, before_window)
        lo = bisect.bisect_left(idx, before_window - lookback)
        if hi - lo < MIN_BASELINE_WINDOWS:
            return None
        hist = self._by_service[service]
        return np.fromiter(
            (hist[i].features.get(metric, 0.0) for i in range(lo, hi)),
            dtype=float,
            count=hi - lo,
        )

    def baseline(
        self,
        service: str,
        metric: str,
        before_window: int,
        lookback: int = BASELINE_SHORT_WINDOWS,
    ) -> Optional[Tuple[float, float, int]]:
        """(mean, std, n) over the windows immediately preceding
        `before_window`. The current window is deliberately excluded so an
        ongoing anomaly cannot inflate its own baseline."""
        arr = self._lookback_values(service, metric, before_window, lookback)
        if arr is None:
            return None
        return float(arr.mean()), float(arr.std()), len(arr)

    def robust_baseline(
        self,
        service: str,
        metric: str,
        before_window: int,
        lookback: int = BASELINE_SHORT_WINDOWS,
    ) -> Optional[Tuple[float, float, int]]:
        """Median / scaled-MAD version, preferred by the z-score detector: a
        couple of already-anomalous windows inside the lookback shift the mean
        a long way but barely move the median."""
        arr = self._lookback_values(service, metric, before_window, lookback)
        if arr is None:
            return None
        median = float(np.median(arr))
        mad = float(np.median(np.abs(arr - median)))
        # 1.4826 makes MAD a consistent estimator of sigma for normal data.
        scale = mad * 1.4826
        if scale <= 0:
            scale = float(arr.std())
        return median, scale, len(arr)

    def normal_matrix(self, columns: List[str], exclude_windows: int = 0) -> np.ndarray:
        """Stack every retained window into a matrix, for unsupervised fits."""
        rows = []
        for service in self.services:
            h = self._by_service[service]
            if exclude_windows:
                h = h[:-exclude_windows] if exclude_windows < len(h) else []
            rows.extend(w.vector(columns) for w in h)
        return np.vstack(rows) if rows else np.empty((0, len(columns)))

    def __len__(self) -> int:
        return sum(len(v) for v in self._by_service.values())
