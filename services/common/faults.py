"""Runtime fault injection for the real services.

Each service exposes `POST /fault/{name}` and `DELETE /fault`. The controller
holds the injected state; the service's own code paths consult it. This is the
same catalogue as `incidentdna.sim.faults`, implemented for real:

    database_latency       a sleep inside the query path (or pg_sleep)
    connection_exhaustion  a semaphore standing in for a smaller pool
    memory_leak            a module-level list that is never emptied
    payment_timeout        a sleep inside the payment handler
    cpu_saturation         a busy loop on a worker thread
    bad_deployment         a version bump plus a failure rate for some inputs
"""
from __future__ import annotations

import asyncio
import logging
import math
import os
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from fastapi import HTTPException
from pydantic import BaseModel, Field

log = logging.getLogger("incidentdna.faults")

FAULT_NAMES = [
    "database_latency",
    "connection_exhaustion",
    "memory_leak",
    "payment_timeout",
    "cpu_saturation",
    "bad_deployment",
]


@dataclass
class ActiveFault:
    name: str
    severity: float
    started_at: float
    duration_seconds: float

    @property
    def elapsed(self) -> float:
        return time.time() - self.started_at

    @property
    def expired(self) -> bool:
        return self.elapsed > self.duration_seconds

    @property
    def ramp(self) -> float:
        """Match the simulator's trapezoid so injected faults look alike in
        both deployments: 18% ramp up, plateau, 12% ramp down."""
        if self.duration_seconds <= 0:
            return 0.0
        p = self.elapsed / self.duration_seconds
        if p <= 0 or p >= 1:
            return 0.0
        if p < 0.18:
            return p / 0.18
        if p > 0.88:
            return max(0.0, (1.0 - p) / 0.12)
        return 1.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "severity": round(self.severity, 3),
            "elapsed_seconds": round(self.elapsed, 1),
            "remaining_seconds": round(max(0.0, self.duration_seconds - self.elapsed), 1),
            "intensity": round(self.ramp, 3),
        }


class FaultController:
    def __init__(self, telemetry=None) -> None:
        self.telemetry = telemetry
        self.active: Dict[str, ActiveFault] = {}
        self._leak: List[bytes] = []
        self._cpu_thread: Optional[threading.Thread] = None
        self._cpu_stop = threading.Event()
        self._lock = threading.Lock()

    # -- control -----------------------------------------------------------
    def inject(self, name: str, severity: float = 0.8, duration_seconds: float = 180.0) -> dict:
        if name not in FAULT_NAMES:
            raise KeyError(f"unknown fault '{name}'; known: {', '.join(FAULT_NAMES)}")
        with self._lock:
            self.active[name] = ActiveFault(name, severity, time.time(), duration_seconds)
        log.warning("fault injected: %s severity=%.2f for %.0fs", name, severity, duration_seconds)
        if name == "cpu_saturation":
            self._start_cpu_burn()
        if name == "bad_deployment" and self.telemetry is not None:
            self.telemetry.emit_deployment("v2.0.0")
        return self.active[name].to_dict()

    def clear(self, name: Optional[str] = None) -> int:
        with self._lock:
            n = len(self.active) if name is None else int(name in self.active)
            if name is None:
                self.active.clear()
            else:
                self.active.pop(name, None)
        self._cpu_stop.set()
        self._leak.clear()
        return n

    def _reap(self) -> None:
        with self._lock:
            for key in [k for k, v in self.active.items() if v.expired]:
                self.active.pop(key)
                log.info("fault expired: %s", key)
        if "cpu_saturation" not in self.active:
            self._cpu_stop.set()
        if "memory_leak" not in self.active and self._leak:
            self._leak.clear()
            if self.telemetry is not None:
                self.telemetry.extra_memory_mb = 0.0

    def get(self, name: str) -> Optional[ActiveFault]:
        self._reap()
        return self.active.get(name)

    def status(self) -> dict:
        self._reap()
        return {
            "active": [f.to_dict() for f in self.active.values()],
            "available": FAULT_NAMES,
        }

    # -- effects the service code calls into -------------------------------
    async def database_delay(self) -> float:
        """Extra seconds a query should take. Applied by the DB caller."""
        extra = 0.0
        f = self.get("database_latency")
        if f:
            extra += f.ramp * (0.05 + 1.2 * f.severity)
        f = self.get("connection_exhaustion")
        if f:
            wait = f.ramp * (0.15 + 1.4 * f.severity)
            extra += wait
            if self.telemetry is not None:
                self.telemetry.pool_utilization = min(0.995, 0.42 + f.ramp * 0.55)
                self.telemetry.pool_wait_ms = wait * 1000.0
        elif self.telemetry is not None:
            self.telemetry.pool_utilization = 0.35
            self.telemetry.pool_wait_ms = 0.0
        if extra > 0:
            await asyncio.sleep(extra)
        return extra

    async def payment_delay(self) -> float:
        f = self.get("payment_timeout")
        if not f:
            return 0.0
        extra = f.ramp * (0.2 + 2.5 * f.severity)
        await asyncio.sleep(extra)
        return extra

    def leak_memory(self) -> None:
        f = self.get("memory_leak")
        if not f:
            return
        # ~1 MB per request retained, with an OOM-style restart at the limit.
        self._leak.append(b"\0" * 1_000_000)
        leaked_mb = len(self._leak)
        if self.telemetry is not None:
            self.telemetry.extra_memory_mb = leaked_mb
        if leaked_mb > 400 + 400 * (1 - f.severity):
            self._leak.clear()
            if self.telemetry is not None:
                self.telemetry.extra_memory_mb = 0.0
                self.telemetry.record_restart()
                self.telemetry.emit_log("ERROR", "oom_killed", heap_mb=leaked_mb)
            log.error("simulated OOM restart after %d MB", leaked_mb)

    def should_fail_request(self, key: str = "") -> bool:
        f = self.get("bad_deployment")
        if not f or f.ramp <= 0:
            return False
        rate = 0.04 + 0.30 * f.severity
        return random.random() < rate

    def _start_cpu_burn(self) -> None:
        if self._cpu_thread is not None and self._cpu_thread.is_alive():
            return
        self._cpu_stop = threading.Event()

        def burn() -> None:
            while not self._cpu_stop.is_set():
                f = self.get("cpu_saturation")
                if f is None:
                    break
                # Duty cycle proportional to severity: busy, then yield.
                busy = 0.02 * (0.4 + 0.6 * f.severity) * max(0.05, f.ramp)
                end = time.perf_counter() + busy
                x = 0.0
                while time.perf_counter() < end:
                    x += math.sqrt(random.random() + 1.0)
                time.sleep(max(0.001, 0.02 - busy))

        self._cpu_thread = threading.Thread(target=burn, name="cpu-burn", daemon=True)
        self._cpu_thread.start()


class InjectBody(BaseModel):
    """Body of `POST /fault/{name}`.

    Declared at module level on purpose: `from __future__ import annotations`
    turns route annotations into strings, and FastAPI resolves them against
    the *module* namespace. A model defined inside the registration function
    is invisible there, and the parameter silently degrades to a query arg.
    """

    severity: float = Field(0.8, ge=0.05, le=1.0)
    duration_seconds: float = Field(180.0, ge=10.0, le=3600.0)


def register_fault_routes(app, controller: FaultController) -> None:
    """Attach `/fault` routes to a FastAPI app."""

    @app.get("/fault")
    def fault_status() -> dict:
        return controller.status()

    @app.post("/fault/{name}")
    def fault_inject(name: str, body: InjectBody) -> dict:
        try:
            return controller.inject(name, body.severity, body.duration_seconds)
        except KeyError as exc:
            raise HTTPException(400, str(exc))

    @app.delete("/fault")
    def fault_clear(name: Optional[str] = None) -> dict:
        return {"cleared": controller.clear(name)}
