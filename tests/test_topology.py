"""The dependency graph is load-bearing: RCA reasoning is defined on it."""
import networkx as nx
import pytest

from incidentdna import topology as topo


def test_graph_is_a_dag():
    assert nx.is_directed_acyclic_graph(topo.graph())


def test_every_service_is_reachable_from_an_entrypoint():
    for service in topo.SERVICES:
        assert topo.distance_from_entrypoint(service) < len(topo.SERVICES)


def test_reverse_topological_order_puts_callees_first():
    order = topo.reverse_topological_order()
    position = {s: i for i, s in enumerate(order)}
    for edge in topo.EDGES:
        assert position[edge.callee] < position[edge.caller], (
            f"{edge.callee} must be resolvable before {edge.caller}"
        )


def test_postgres_is_a_leaf_and_the_gateway_is_a_root():
    assert topo.callees(topo.POSTGRES) == []
    assert topo.callers(topo.API_GATEWAY) == []


def test_inventory_has_two_callers():
    # The one node with fan-in; propagation-consistency logic depends on it.
    assert set(topo.callers(topo.INVENTORY_SERVICE)) == {
        topo.API_GATEWAY, topo.ORDER_SERVICE
    }


def test_transitive_relations_are_consistent():
    for service in topo.SERVICES:
        for caller in topo.transitive_callers(service):
            assert service in topo.transitive_callees(caller)


def test_serialisable_topology_matches_the_graph():
    payload = topo.as_dict()
    assert len(payload["services"]) == len(topo.SERVICES)
    assert len(payload["edges"]) == len(topo.EDGES)
