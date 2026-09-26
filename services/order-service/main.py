"""Order Service — coordinates inventory, payment and persistence.

The only service with fan-out, so it is where symptoms of three different
causes converge, and where distinguishing "I am slow" from "I am waiting"
actually matters.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fastapi import HTTPException
from pydantic import BaseModel, Field

from incidentdna import topology as topo
from services.common.db import Database
from services.common.service import create_service


class CheckoutRequest(BaseModel):
    user_id: int = Field(1, ge=1)
    product_id: int = Field(1, ge=1)
    quantity: int = Field(1, ge=1, le=20)


async def _startup(ctx) -> None:
    ctx.db = Database(ctx.telemetry, ctx.faults)
    await ctx.db.connect()


async def _shutdown(ctx) -> None:
    await ctx.db.close()


app, ctx = create_service(
    topo.ORDER_SERVICE, on_startup=_startup, on_shutdown=_shutdown
)


@app.post("/checkout")
async def checkout(req: CheckoutRequest) -> dict:
    # A memory leak is only observable under traffic, so it is driven by the
    # request path rather than by a timer.
    ctx.faults.leak_memory()
    if ctx.faults.should_fail_request():
        ctx.telemetry.emit_log("ERROR", "unhandled_exception", endpoint="/checkout")
        raise HTTPException(500, "order regression")

    inventory = await ctx.call(
        topo.INVENTORY_SERVICE, f"/check/{req.product_id}?quantity={req.quantity}"
    )
    if inventory is None:
        raise HTTPException(503, "inventory check failed")
    if not inventory.get("available"):
        raise HTTPException(409, "out of stock")

    total = inventory["price_cents"] * req.quantity
    payment = await ctx.call(
        topo.PAYMENT_SERVICE,
        "/authorize",
        method="POST",
        json={"user_id": req.user_id, "amount_cents": total},
    )
    if payment is None:
        raise HTTPException(504, "payment authorisation timed out")
    if not payment.get("authorized"):
        raise HTTPException(402, "payment declined")

    try:
        order_id = await ctx.db.insert_order(req.user_id, req.product_id, req.quantity, total)
    except Exception:
        raise HTTPException(504, "order persistence failed")

    return {
        "order_id": order_id,
        "user_id": req.user_id,
        "product_id": req.product_id,
        "quantity": req.quantity,
        "total_cents": total,
        "authorization_id": payment.get("authorization_id"),
    }


@app.get("/orders/{order_id}")
async def get_order(order_id: int) -> dict:
    try:
        order = await ctx.db.get_order(order_id)
    except Exception:
        raise HTTPException(504, "order lookup failed")
    if order is None:
        raise HTTPException(404, "no such order")
    return order
