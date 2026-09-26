import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from incidentdna.config import SimConfig
from incidentdna.pipeline.analyzer import Analyzer
from incidentdna.sim.engine import SimulationEngine


@pytest.fixture(scope="session")
def sim_config() -> SimConfig:
    return SimConfig(run_windows=96)


def run_fault(fault_type, seed=11, severity=0.85, origin=None,
              start_window=45, duration_windows=32, config=None, detector=None):
    """Simulate one run with one injected fault and analyse it end to end.

    Returns (analyzer, active_fault, engine)."""
    cfg = config or SimConfig(run_windows=96)
    engine = SimulationEngine(seed=seed, config=cfg, run_id="t")
    fault = None
    if fault_type is not None:
        fault = engine.inject(fault_type, origin=origin, severity=severity,
                              start_window=start_window, duration_windows=duration_windows)
    analyzer = Analyzer(detector=detector, run_id="t")
    for _ in range(cfg.run_windows):
        analyzer.ingest(engine.step())
    analyzer.finish()
    return analyzer, fault, engine
