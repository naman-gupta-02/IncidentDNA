"""FastAPI incident service.

Serves the investigation dashboard and a JSON API over the live system. Every
endpoint reads from the same `Analyzer` that the offline evaluation uses, so
what you see in the browser is produced by the evaluated code path.

    uvicorn backend.app:app --reload --port 8000

Environment:
    INCIDENTDNA_SOURCE     `simulated` (default) or `kafka`
    INCIDENTDNA_SPEED      simulated windows per real window (default 10)
    INCIDENTDNA_SEED       simulator seed
    KAFKA_BOOTSTRAP_SERVERS  when the source is `kafka`
"""
from __future__ import annotations

import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.live import LiveSystem
from incidentdna import topology as topo
from incidentdna.config import EXPERIMENTS_DIR, WINDOW_SECONDS
from incidentdna.sim import faults as fault_lib

logging.basicConfig(
    level=os.environ.get("INCIDENTDNA_LOG", "INFO"),
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
log = logging.getLogger("incidentdna.api")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
RESULTS_DIR = EXPERIMENTS_DIR / "results"

system: Optional[LiveSystem] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global system
    system = LiveSystem(
        source_kind=os.environ.get("INCIDENTDNA_SOURCE", "simulated"),
        speed=float(os.environ.get("INCIDENTDNA_SPEED", "10")),
        seed=int(os.environ.get("INCIDENTDNA_SEED", "7")),
        # Set INCIDENTDNA_CAPTURE to tee every record to a JSONL file that
        # `scripts/replay.py` can feed back through the same pipeline.
        capture_path=os.environ.get("INCIDENTDNA_CAPTURE"),
    )
    system.start()
    yield
    system.stop()


app = FastAPI(
    title="IncidentDNA",
    description="AI-powered production incident investigator",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def live() -> LiveSystem:
    if system is None:  # pragma: no cover - only before startup
        raise HTTPException(503, "live system not started")
    return system


# --- meta -----------------------------------------------------------------
@app.get("/api/health")
def health() -> dict:
    s = live()
    return {
        "status": "ok",
        "source": s.source_kind,
        "windows_processed": s.analyzer.processed_windows,
        "incidents": len(s.analyzer.incidents),
    }


@app.get("/api/topology")
def get_topology() -> dict:
    return topo.as_dict()


@app.get("/api/faults")
def get_faults() -> List[dict]:
    return fault_lib.catalogue()


# --- live state -----------------------------------------------------------
@app.get("/api/state")
def get_state() -> dict:
    return live().snapshot()


@app.get("/api/metrics")
def get_metrics(
    services: Optional[str] = Query(None, description="comma-separated; default all"),
    metrics: str = Query("latency_p95,error_rate,self_latency_p95,db_latency_p95,cpu_mean,pool_utilization"),
    limit: int = Query(160, ge=10, le=1000),
) -> dict:
    service_list = services.split(",") if services else list(topo.SERVICES)
    unknown = [s for s in service_list if s not in topo.SERVICES]
    if unknown:
        raise HTTPException(400, f"unknown services: {unknown}")
    return live().metric_series(service_list, metrics.split(","), limit=limit)


@app.get("/api/traces")
def get_traces(incident_id: Optional[str] = None, limit: int = Query(8, ge=1, le=50)) -> List[dict]:
    return live().traces(incident_id=incident_id, limit=limit)


# --- incidents ------------------------------------------------------------
@app.get("/api/incidents")
def list_incidents() -> List[dict]:
    return live().incidents()


@app.get("/api/incidents/{incident_id}")
def get_incident(incident_id: str) -> dict:
    payload = live().incident_payload(incident_id)
    if payload is None:
        raise HTTPException(404, f"no incident {incident_id}")
    return payload


@app.get("/api/incidents/{incident_id}/summary", response_class=PlainTextResponse)
def get_incident_summary(incident_id: str) -> str:
    s = live()
    incident = s.analyzer.manager.by_id(incident_id)
    if incident is None:
        raise HTTPException(404, f"no incident {incident_id}")
    return incident.summary_text(WINDOW_SECONDS)


class RankerRequest(BaseModel):
    ranker: str = Field(..., description="one of /api/state.available_rankers")


@app.post("/api/ranker")
def set_ranker(req: RankerRequest) -> dict:
    s = live()
    try:
        name = s.set_ranker(req.ranker)
    except KeyError:
        raise HTTPException(400, f"unknown ranker '{req.ranker}'; have {sorted(s.rankers)}")
    return {"ranker": name, "rediagnosed": len(s.analyzer.incidents)}


# --- fault injection ------------------------------------------------------
class InjectRequest(BaseModel):
    fault_type: str
    origin: Optional[str] = None
    severity: float = Field(0.85, ge=0.05, le=1.0)
    duration_seconds: float = Field(180.0, ge=30.0, le=1800.0)


@app.post("/api/faults/inject")
def inject_fault(req: InjectRequest) -> dict:
    s = live()
    try:
        record = s.inject(
            req.fault_type,
            origin=req.origin,
            severity=req.severity,
            duration_seconds=req.duration_seconds,
        )
    except KeyError as exc:
        raise HTTPException(400, str(exc))
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(400, str(exc))
    return record.to_dict()


@app.post("/api/faults/clear")
def clear_faults() -> dict:
    return {"cleared": live().clear_faults()}


# --- evaluation results ---------------------------------------------------
@app.get("/api/evaluation")
def get_evaluation() -> dict:
    path = RESULTS_DIR / "report.json"
    if not path.exists():
        raise HTTPException(
            404, "no evaluation report yet; run `make dataset train evaluate`"
        )
    return json.loads(path.read_text())


# --- frontend -------------------------------------------------------------
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "index.html")
