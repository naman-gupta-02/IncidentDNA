"""Inventory Service — stock lookups, backed by an in-process cache.

Called by both the gateway (product pages) and the order service (checkout),
which makes it the one node with two callers: a fault here should show a
propagation pattern the single-caller faults do not.
"""
from __future__ import annotations

import asyncio
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fastapi import HTTPException

from incidentdna import topology as topo
from services.common.service import create_service

app, ctx = create_service(topo.INVENTORY_SERVICE)

CATALOGUE = {
    i: {"id": i, "name": f"product-{i}", "price_cents": 500 + 37 * i, "stock": 500}
    for i in range(1, 21)
}


async def _lookup(product_id: int) -> dict:
    await asyncio.sleep(random.lognormvariate(-4.07, 0.34))  # ~17 ms median
    product = CATALOGUE.get(product_id)
    if product is None:
        raise HTTPException(404, "no such product")
    return product


@app.get("/products/{product_id}")
async def get_product(product_id: int) -> dict:
    if ctx.faults.should_fail_request():
        ctx.telemetry.emit_log("ERROR", "unhandled_exception", endpoint="/products")
        raise HTTPException(500, "inventory regression")
    return await _lookup(product_id)


@app.get("/check/{product_id}")
async def check(product_id: int, quantity: int = 1) -> dict:
    product = await _lookup(product_id)
    return {
        "product_id": product_id,
        "available": product["stock"] >= quantity,
        "stock": product["stock"],
        "price_cents": product["price_cents"],
    }
