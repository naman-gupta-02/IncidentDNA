"""API Gateway — the user-facing entrypoint.

Owns no business logic: it authenticates nothing, stores nothing, and exists
so that the *symptom* of every incident appears here while the *cause* is
somewhere behind it.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fastapi import HTTPException
from pydantic import BaseModel, Field

from incidentdna import topology as topo
from services.common.service import create_service

app, ctx = create_service(topo.API_GATEWAY)


class CheckoutRequest(BaseModel):
    user_id: int = Field(1, ge=1)
    product_id: int = Field(1, ge=1)
    quantity: int = Field(1, ge=1, le=20)


@app.post("/checkout")
async def checkout(req: CheckoutRequest) -> dict:
    if ctx.faults.should_fail_request():
        ctx.telemetry.emit_log("ERROR", "unhandled_exception", endpoint="/checkout")
        raise HTTPException(500, "gateway regression")
    result = await ctx.call(topo.ORDER_SERVICE, "/checkout", method="POST", json=req.model_dump())
    if result is None:
        raise HTTPException(504, "order service unavailable")
    return result


@app.get("/orders/{order_id}")
async def get_order(order_id: int) -> dict:
    result = await ctx.call(topo.ORDER_SERVICE, f"/orders/{order_id}")
    if result is None:
        raise HTTPException(504, "order service unavailable")
    return result


@app.get("/products/{product_id}")
async def get_product(product_id: int) -> dict:
    result = await ctx.call(topo.INVENTORY_SERVICE, f"/products/{product_id}")
    if result is None:
        raise HTTPException(503, "inventory service unavailable")
    return result
