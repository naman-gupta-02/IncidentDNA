"""Payment Service — simulates an external payment provider.

A leaf with no dependencies of its own, so when it degrades every other
affected service is downstream of it. That is the cleanest test of whether the
ranker follows the dependency graph or just the loudest metric.
"""
from __future__ import annotations

import asyncio
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fastapi import HTTPException
from pydantic import BaseModel, Field

from incidentdna import topology as topo
from incidentdna.telemetry.schema import new_id
from services.common.service import create_service

app, ctx = create_service(topo.PAYMENT_SERVICE)

#: Baseline authorisation latency of the upstream provider.
BASE_LATENCY_S = 0.145
DECLINE_RATE = 0.015


class AuthorizeRequest(BaseModel):
    user_id: int = Field(1, ge=1)
    amount_cents: int = Field(1000, ge=1)


@app.post("/authorize")
async def authorize(req: AuthorizeRequest) -> dict:
    if ctx.faults.should_fail_request():
        ctx.telemetry.emit_log("ERROR", "unhandled_exception", endpoint="/authorize")
        raise HTTPException(500, "payment regression")

    await asyncio.sleep(random.lognormvariate(-1.93, 0.42))  # ~145 ms median
    extra = await ctx.faults.payment_delay()
    if extra > 0:
        ctx.telemetry.emit_log("WARN", "provider_slow", added_seconds=round(extra, 3))

    if random.random() < DECLINE_RATE:
        return {"authorized": False, "reason": "declined", "amount_cents": req.amount_cents}
    return {
        "authorized": True,
        "authorization_id": new_id(12),
        "amount_cents": req.amount_cents,
    }
