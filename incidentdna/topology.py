"""The service dependency graph.

    api-gateway ──> order-service ──> payment-service
         │                │──────────> inventory-service
         │                └──────────> postgres
         └────────────> inventory-service

Edges point from caller to callee. A fault at a callee propagates *up* the
edges to its callers, which is what makes root-cause ranking non-trivial:
the user-facing symptom is always at `api-gateway`, never at the cause.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Tuple

import networkx as nx

API_GATEWAY = "api-gateway"
ORDER_SERVICE = "order-service"
PAYMENT_SERVICE = "payment-service"
INVENTORY_SERVICE = "inventory-service"
POSTGRES = "postgres"

SERVICES: List[str] = [
    API_GATEWAY,
    ORDER_SERVICE,
    PAYMENT_SERVICE,
    INVENTORY_SERVICE,
    POSTGRES,
]

ENTRYPOINTS: List[str] = [API_GATEWAY]


@dataclass(frozen=True)
class Edge:
    caller: str
    callee: str
    operation: str
    #: Average number of calls to `callee` per request handled by `caller`.
    fanout: float
    #: Client-side timeout in ms. Exceeding it produces timeouts + retries.
    timeout_ms: float
    #: Retries the caller attempts before giving up.
    retries: int


EDGES: List[Edge] = [
    Edge(API_GATEWAY, ORDER_SERVICE, "POST /checkout", fanout=1.0, timeout_ms=3000, retries=0),
    Edge(API_GATEWAY, INVENTORY_SERVICE, "GET /products/{id}", fanout=0.6, timeout_ms=800, retries=1),
    Edge(ORDER_SERVICE, INVENTORY_SERVICE, "inventory/check", fanout=1.0, timeout_ms=600, retries=1),
    Edge(ORDER_SERVICE, PAYMENT_SERVICE, "payment/authorize", fanout=1.0, timeout_ms=900, retries=1),
    Edge(ORDER_SERVICE, POSTGRES, "INSERT order", fanout=2.0, timeout_ms=1200, retries=0),
]

#: Services that talk to a database. Used for db_latency features.
DATABASE_SERVICES = {POSTGRES}

#: Human-facing display names.
DISPLAY_NAMES: Dict[str, str] = {
    API_GATEWAY: "API Gateway",
    ORDER_SERVICE: "Order Service",
    PAYMENT_SERVICE: "Payment Service",
    INVENTORY_SERVICE: "Inventory Service",
    POSTGRES: "PostgreSQL",
}


@lru_cache(maxsize=1)
def graph() -> nx.DiGraph:
    """Directed graph with edges caller -> callee."""
    g = nx.DiGraph()
    g.add_nodes_from(SERVICES)
    for e in EDGES:
        g.add_edge(
            e.caller,
            e.callee,
            operation=e.operation,
            fanout=e.fanout,
            timeout_ms=e.timeout_ms,
            retries=e.retries,
        )
    return g


def callees(service: str) -> List[str]:
    """Services that `service` calls (its dependencies)."""
    return list(graph().successors(service))


def callers(service: str) -> List[str]:
    """Services that call `service` (its dependents / downstream symptoms)."""
    return list(graph().predecessors(service))


def edge(caller: str, callee: str) -> Edge:
    for e in EDGES:
        if e.caller == caller and e.callee == callee:
            return e
    raise KeyError(f"no edge {caller} -> {callee}")


@lru_cache(maxsize=1)
def reverse_topological_order() -> List[str]:
    """Callees before callers, so a window can be resolved in one pass."""
    return list(reversed(list(nx.topological_sort(graph()))))


@lru_cache(maxsize=None)
def transitive_callers(service: str) -> frozenset:
    """Everything that can observe `service` degrading, directly or not."""
    return frozenset(nx.ancestors(graph(), service))


@lru_cache(maxsize=None)
def transitive_callees(service: str) -> frozenset:
    """Everything `service` depends on, directly or not."""
    return frozenset(nx.descendants(graph(), service))


@lru_cache(maxsize=None)
def distance_from_entrypoint(service: str) -> int:
    """Shortest hop count from a user-facing entrypoint. Entrypoints are 0."""
    g = graph()
    best = None
    for ep in ENTRYPOINTS:
        if ep == service:
            return 0
        try:
            d = nx.shortest_path_length(g, ep, service)
        except nx.NetworkXNoPath:
            continue
        best = d if best is None else min(best, d)
    return best if best is not None else len(SERVICES)


def call_paths() -> List[Tuple[str, ...]]:
    """All root-to-leaf call paths, used to build synthetic traces."""
    g = graph()
    paths: List[Tuple[str, ...]] = []
    for ep in ENTRYPOINTS:
        for node in SERVICES:
            if node == ep:
                continue
            for p in nx.all_simple_paths(g, ep, node):
                paths.append(tuple(p))
    return paths


def as_dict() -> dict:
    """Serialisable topology for the dashboard."""
    return {
        "services": [
            {
                "name": s,
                "display_name": DISPLAY_NAMES[s],
                "is_entrypoint": s in ENTRYPOINTS,
                "is_database": s in DATABASE_SERVICES,
                "depth": distance_from_entrypoint(s),
            }
            for s in SERVICES
        ],
        "edges": [
            {
                "source": e.caller,
                "target": e.callee,
                "operation": e.operation,
                "timeout_ms": e.timeout_ms,
                "retries": e.retries,
            }
            for e in EDGES
        ],
    }


def validate() -> None:
    g = graph()
    if not nx.is_directed_acyclic_graph(g):
        raise ValueError("service dependency graph must be acyclic")
    unreachable = [s for s in SERVICES if distance_from_entrypoint(s) >= len(SERVICES)]
    if unreachable:
        raise ValueError(f"services unreachable from an entrypoint: {unreachable}")


validate()
