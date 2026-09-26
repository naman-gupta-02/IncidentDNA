"""Database access plus PostgreSQL telemetry.

PostgreSQL cannot run our agent, so its telemetry is produced by client-side
instrumentation in the service that queries it and attributed to the
`postgres` service. That is the normal arrangement for managed databases, and
it is what lets the ranker treat Postgres as a first-class candidate:

* every query emits a **client span** on the calling service and a **server
  span attributed to `postgres`**, so trace evidence can credit the database
  for the wall-clock time it actually held;
* query duration, connection-pool utilisation and error counts are emitted as
  `postgres` metrics.

With `DATABASE_URL` set the queries are real (asyncpg). Without it the module
falls back to an in-process store with representative latency, so the services
can be run and exercised without a database.
"""
from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from typing import Any, Dict, List, Optional

from incidentdna import topology as topo
from incidentdna.pipeline.features import (
    M_DB_DURATION,
    M_DURATION,
    M_ERRORS,
    M_POOL_UTIL,
    M_POOL_WAIT,
    M_REQUESTS,
    M_SELF_DURATION,
)
from incidentdna.telemetry.schema import (
    MetricPoint,
    SPAN_KIND_SERVER,
    Span,
    TelemetryBatch,
    new_id,
)

from .telemetry import Telemetry, _current_span, _current_trace, _sampled

log = logging.getLogger("incidentdna.db")

POOL_SIZE = int(os.environ.get("DB_POOL_SIZE", "10"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS products (
    id SERIAL PRIMARY KEY, name TEXT NOT NULL, price_cents INT NOT NULL, stock INT NOT NULL
);
CREATE TABLE IF NOT EXISTS orders (
    id SERIAL PRIMARY KEY, user_id INT NOT NULL, product_id INT NOT NULL,
    quantity INT NOT NULL, total_cents INT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_orders_user_id ON orders(user_id);
"""


class Database:
    """Query interface that emits `postgres` telemetry for every call."""

    def __init__(self, telemetry: Telemetry, faults) -> None:
        self.telemetry = telemetry
        self.faults = faults
        self.pool = None
        self.dsn = os.environ.get("DATABASE_URL")
        self._memory_orders: List[dict] = []
        self._memory_products = {
            i: {"id": i, "name": f"product-{i}", "price_cents": 500 + 37 * i, "stock": 500}
            for i in range(1, 21)
        }
        self._active = 0
        self._counters = {"requests": 0.0, "errors": 0.0}
        self._semaphore = asyncio.Semaphore(POOL_SIZE)

    # -- lifecycle ---------------------------------------------------------
    async def connect(self) -> None:
        if not self.dsn:
            log.warning("DATABASE_URL unset; using the in-process store")
            return
        try:
            import asyncpg  # type: ignore

            self.pool = await asyncpg.create_pool(self.dsn, min_size=2, max_size=POOL_SIZE)
            async with self.pool.acquire() as conn:
                await conn.execute(SCHEMA)
                count = await conn.fetchval("SELECT count(*) FROM products")
                if not count:
                    await conn.executemany(
                        "INSERT INTO products(name, price_cents, stock) VALUES($1,$2,$3)",
                        [(f"product-{i}", 500 + 37 * i, 500) for i in range(1, 21)],
                    )
            log.info("connected to postgres (pool=%d)", POOL_SIZE)
        except Exception as exc:
            log.warning("postgres unavailable (%s); using the in-process store", exc)
            self.pool = None

    async def close(self) -> None:
        if self.pool is not None:
            await self.pool.close()

    # -- instrumentation ---------------------------------------------------
    def _emit(self, name: str, value: float, kind: str = "gauge") -> None:
        with self.telemetry._lock:
            self.telemetry._pending.metrics.append(
                MetricPoint(time.time(), topo.POSTGRES, name, float(value), kind, "pg16")
            )

    def flush_counters(self) -> None:
        """Called on the service's flush tick so postgres gets counters too."""
        with self.telemetry._lock:
            c, self._counters = self._counters, {"requests": 0.0, "errors": 0.0}
        self._emit(M_REQUESTS, c["requests"], "counter")
        self._emit(M_ERRORS, c["errors"], "counter")
        self._emit(M_POOL_UTIL, min(1.0, self._active / POOL_SIZE))

    async def query(self, operation: str, sql: str, *args, fetch: str = "all") -> Any:
        """Run a query, emitting a client span on this service and a server
        span attributed to `postgres`."""
        edge = topo.edge(self.telemetry.service, topo.POSTGRES)
        parent = _current_span.get()
        started = time.time()
        failed = False
        self._active += 1
        self._counters["requests"] += 1

        pg_span = Span(
            trace_id=_current_trace.get() or new_id(32),
            span_id=new_id(16),
            parent_span_id=None,
            service=topo.POSTGRES,
            operation=operation,
            kind=SPAN_KIND_SERVER,
            start_time=started,
            duration_ms=0.0,
            attributes={"db.system": "postgresql", "db.statement": operation},
        )
        try:
            with self.telemetry.client_span(topo.POSTGRES, operation, edge.timeout_ms, edge.retries) as client:
                pg_span.parent_span_id = client.span_id
                try:
                    async with self._semaphore:
                        await self.faults.database_delay()
                        result = await asyncio.wait_for(
                            self._execute(sql, *args, fetch=fetch), timeout=edge.timeout_ms / 1000.0
                        )
                    client.status = "OK"
                    return result
                except asyncio.TimeoutError:
                    client.status = "TIMEOUT"
                    pg_span.status = "TIMEOUT"
                    failed = True
                    raise
                except Exception:
                    client.status = "ERROR"
                    pg_span.status = "ERROR"
                    failed = True
                    raise
        finally:
            self._active -= 1
            duration_ms = (time.time() - started) * 1000.0
            pg_span.duration_ms = duration_ms
            if failed:
                self._counters["errors"] += 1
            if _sampled.get():
                self.telemetry.emit_span(pg_span)
            # postgres-attributed latency, and the caller's db-time metric.
            self._emit(M_DURATION, duration_ms, "histogram")
            self._emit(M_SELF_DURATION, duration_ms, "histogram")
            self._emit(M_DB_DURATION, duration_ms, "histogram")
            self.telemetry.record_db_query(duration_ms)
            if self.telemetry.pool_wait_ms:
                self._emit(M_POOL_WAIT, self.telemetry.pool_wait_ms)

    async def _execute(self, sql: str, *args, fetch: str = "all") -> Any:
        if self.pool is None:
            return await self._execute_memory(sql, *args, fetch=fetch)
        async with self.pool.acquire() as conn:
            if fetch == "val":
                return await conn.fetchval(sql, *args)
            if fetch == "row":
                row = await conn.fetchrow(sql, *args)
                return dict(row) if row else None
            return [dict(r) for r in await conn.fetch(sql, *args)]

    async def _execute_memory(self, sql: str, *args, fetch: str = "all") -> Any:
        """In-process stand-in with representative latency."""
        await asyncio.sleep(random.lognormvariate(-5.0, 0.5))  # ~7 ms median
        lowered = sql.strip().lower()
        if lowered.startswith("insert into orders"):
            order = {
                "id": len(self._memory_orders) + 1,
                "user_id": args[0],
                "product_id": args[1],
                "quantity": args[2],
                "total_cents": args[3],
            }
            self._memory_orders.append(order)
            return order["id"] if fetch == "val" else order
        if "from orders" in lowered:
            oid = args[0] if args else None
            found = next((o for o in self._memory_orders if o["id"] == oid), None)
            return found if fetch == "row" else ([found] if found else [])
        if "from products" in lowered:
            pid = args[0] if args else 1
            product = self._memory_products.get(pid)
            return product if fetch == "row" else list(self._memory_products.values())
        return None if fetch in ("row", "val") else []

    # -- domain helpers ----------------------------------------------------
    async def get_product(self, product_id: int) -> Optional[dict]:
        return await self.query(
            "SELECT products", "SELECT * FROM products WHERE id = $1", product_id, fetch="row"
        )

    async def get_order(self, order_id: int) -> Optional[dict]:
        return await self.query(
            "SELECT orders", "SELECT * FROM orders WHERE id = $1", order_id, fetch="row"
        )

    async def insert_order(self, user_id: int, product_id: int, quantity: int, total: int) -> Any:
        return await self.query(
            "INSERT order",
            "INSERT INTO orders(user_id, product_id, quantity, total_cents) "
            "VALUES($1,$2,$3,$4) RETURNING id",
            user_id,
            product_id,
            quantity,
            total,
            fetch="val",
        )
