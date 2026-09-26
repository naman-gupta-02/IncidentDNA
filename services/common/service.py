"""Factory for an instrumented microservice.

Every service in `services/` is the same shell — telemetry middleware, fault
routes, a health check and an outbound HTTP client that respects the timeout
and retry budget declared on the dependency graph — plus its own handlers.
Keeping the shell here is what makes the four services genuinely comparable
and keeps their business logic as small as the guide asks for.
"""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Callable, Dict, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from incidentdna import topology as topo

from .faults import FaultController, register_fault_routes
from .telemetry import Telemetry

log = logging.getLogger("incidentdna.service")

#: Where each service can be reached inside the compose network.
DEFAULT_ENDPOINTS: Dict[str, str] = {
    topo.API_GATEWAY: "http://api-gateway:8000",
    topo.ORDER_SERVICE: "http://order-service:8000",
    topo.PAYMENT_SERVICE: "http://payment-service:8000",
    topo.INVENTORY_SERVICE: "http://inventory-service:8000",
}


def endpoint_for(service: str) -> str:
    env = f"{service.replace('-', '_').upper()}_URL"
    return os.environ.get(env, DEFAULT_ENDPOINTS.get(service, "http://localhost:8000"))


class ServiceContext:
    """Handed to each service's route handlers."""

    def __init__(self, name: str, version: str) -> None:
        self.name = name
        self.version = version
        self.telemetry = Telemetry(name, version)
        self.faults = FaultController(self.telemetry)
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def call(
        self, peer: str, path: str, method: str = "GET", json: Any = None
    ) -> Optional[dict]:
        """Call a dependency, honouring the timeout and retry budget declared
        on the edge, and recording a client span either way.

        Returns the decoded body, or None when every attempt failed. The
        caller decides whether that is fatal — which is what produces the
        difference between a timeout and a 5xx in the telemetry."""
        edge = topo.edge(self.name, peer)
        url = endpoint_for(peer).rstrip("/") + path
        timeout_s = edge.timeout_ms / 1000.0
        attempts = edge.retries + 1

        with self.telemetry.client_span(peer, path, edge.timeout_ms, edge.retries) as span:
            for attempt in range(attempts):
                if attempt:
                    self.telemetry.record_retry()
                try:
                    headers = {}
                    tp = self.telemetry.traceparent()
                    if tp:
                        headers["traceparent"] = tp
                    response = await self.client.request(
                        method, url, json=json, headers=headers, timeout=timeout_s
                    )
                    if response.status_code >= 500:
                        span.status = "ERROR"
                        continue
                    span.status = "OK"
                    return response.json()
                except (httpx.TimeoutException, httpx.TransportError):
                    span.status = "TIMEOUT"
            self.telemetry.emit_log(
                "ERROR",
                _failure_event(peer),
                dependency=peer,
                attempts=attempts,
                timeout_ms=edge.timeout_ms,
            )
            return None


def _failure_event(peer: str) -> str:
    return {
        topo.POSTGRES: "database_timeout",
        topo.PAYMENT_SERVICE: "payment_timeout",
        topo.INVENTORY_SERVICE: "inventory_unavailable",
        topo.ORDER_SERVICE: "upstream_timeout",
    }.get(peer, "dependency_failed")


def create_service(
    name: str,
    version: Optional[str] = None,
    on_startup: Optional[Callable[[ServiceContext], Any]] = None,
    on_shutdown: Optional[Callable[[ServiceContext], Any]] = None,
) -> tuple:
    """(app, ctx). Register routes on `app`, read state from `ctx`."""
    version = version or os.environ.get("SERVICE_VERSION", "v1.4.0")
    ctx = ServiceContext(name, version)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if on_startup:
            result = on_startup(ctx)
            if asyncio.iscoroutine(result):
                await result
        log.info("%s %s started", name, version)
        yield
        if on_shutdown:
            result = on_shutdown(ctx)
            if asyncio.iscoroutine(result):
                await result
        ctx.telemetry.close()
        if ctx._client is not None:
            await ctx._client.aclose()

    app = FastAPI(title=name, version=version, lifespan=lifespan)

    @app.middleware("http")
    async def instrument(request: Request, call_next):
        # Operational endpoints are not part of the measured workload.
        if request.url.path.startswith(("/fault", "/health", "/docs", "/openapi")):
            return await call_next(request)
        operation = f"{request.method} {request.scope.get('route').path if request.scope.get('route') else request.url.path}"
        with ctx.telemetry.server_span(operation, request.headers.get("traceparent")) as span:
            try:
                response = await call_next(request)
            except Exception:
                span.status = "ERROR"
                raise
            if response.status_code >= 500:
                span.status = "ERROR"
            span.attributes["http.status_code"] = response.status_code
            response.headers["x-trace-id"] = span.trace_id
            return response

    @app.get("/health")
    def health() -> dict:
        return {"service": name, "version": ctx.telemetry.version, "status": "ok"}

    register_fault_routes(app, ctx.faults)
    return app, ctx
