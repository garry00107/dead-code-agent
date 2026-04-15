"""
Tests for GraphTraversal — verifies SCC algorithm and zombie detection.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph_builder import CodeGraph, GraphNode
from graph_traversal import GraphTraversal, find_zombies


# ─── Helpers ────────────────────────────────────────────────────────

def _make_graph(nodes: list[str], edges: list[tuple[str, str]]) -> CodeGraph:
    """Create a simple CodeGraph from node names and edge pairs."""
    graph = CodeGraph()
    for name in nodes:
        graph.add_node(GraphNode(
            fqn=name,
            file_path=f"{name}.py",
            start_line=1,
            end_line=10,
            kind="function",
            name=name.split(".")[-1],
        ))
    for src, dst in edges:
        graph.add_edge(src, dst)
    return graph


# ─── Alive Set Computation ────────────────────────────────────────

class TestAliveSet:
    def test_root_is_alive(self):
        graph = _make_graph(["main", "helper"], [("main", "helper")])
        traversal = GraphTraversal(graph, roots={"main"})
        alive = traversal._compute_alive_set()
        assert "main" in alive

    def test_reachable_from_root_is_alive(self):
        graph = _make_graph(["main", "a", "b"], [("main", "a"), ("a", "b")])
        traversal = GraphTraversal(graph, roots={"main"})
        alive = traversal._compute_alive_set()
        assert "a" in alive
        assert "b" in alive

    def test_unreachable_is_dead(self):
        graph = _make_graph(["main", "alive", "dead"], [("main", "alive")])
        traversal = GraphTraversal(graph, roots={"main"})
        alive = traversal._compute_alive_set()
        assert "dead" not in alive

    def test_glob_roots(self):
        graph = _make_graph(["src.api.handler", "src.internal.secret"], [])
        traversal = GraphTraversal(graph, roots={"src.api.*"})
        alive = traversal._compute_alive_set()
        assert "src.api.handler" in alive
        assert "src.internal.secret" not in alive


# ─── SCC Detection ────────────────────────────────────────────────

class TestSCCDetection:
    def test_mutual_recursion_is_one_scc(self):
        # A calls B, B calls A — both dead, one SCC
        graph = _make_graph(["main", "a", "b"], [("a", "b"), ("b", "a")])
        traversal = GraphTraversal(graph, roots={"main"})
        sccs = traversal._tarjan_scc({"a", "b"})
        # Should find one SCC containing both a and b
        assert len(sccs) == 1
        assert set(sccs[0]) == {"a", "b"}

    def test_isolated_node_is_own_scc(self):
        graph = _make_graph(["main", "orphan"], [])
        traversal = GraphTraversal(graph, roots={"main"})
        sccs = traversal._tarjan_scc({"orphan"})
        assert len(sccs) == 1
        assert sccs[0] == ["orphan"]

    def test_chain_produces_separate_sccs(self):
        # a → b → c (no cycles) — each is its own SCC
        graph = _make_graph(["root", "a", "b", "c"], [("a", "b"), ("b", "c")])
        traversal = GraphTraversal(graph, roots={"root"})
        sccs = traversal._tarjan_scc({"a", "b", "c"})
        assert len(sccs) == 3  # Each node is its own SCC


# ─── Zombie Cluster Detection ────────────────────────────────────

class TestZombieClusters:
    def test_simple_dead_code(self):
        graph = _make_graph(
            ["main", "alive_helper", "dead_func"],
            [("main", "alive_helper")],
        )
        clusters = find_zombies(graph, roots={"main"})
        assert len(clusters) == 1
        assert clusters[0].symbols[0].fqn == "dead_func"

    def test_mutual_recursion_zombie(self):
        # A and B only call each other — both dead
        graph = _make_graph(
            ["main", "alive", "zombie_a", "zombie_b"],
            [("main", "alive"), ("zombie_a", "zombie_b"), ("zombie_b", "zombie_a")],
        )
        clusters = find_zombies(graph, roots={"main"})
        assert len(clusters) == 1
        fqns = {s.fqn for s in clusters[0].symbols}
        assert fqns == {"zombie_a", "zombie_b"}

    def test_no_zombies_when_all_reachable(self):
        graph = _make_graph(
            ["main", "a", "b"],
            [("main", "a"), ("a", "b")],
        )
        clusters = find_zombies(graph, roots={"main"})
        assert len(clusters) == 0

    def test_cluster_has_deterministic_id(self):
        graph = _make_graph(["root", "dead"], [])
        clusters = find_zombies(graph, roots={"root"})
        assert len(clusters) == 1
        assert len(clusters[0].cluster_id) == 16  # SHA256 truncated

    def test_clusters_sorted_by_size(self):
        # Create two dead clusters of different sizes
        graph = _make_graph(
            ["root", "small_dead", "big_a", "big_b", "big_c"],
            [("big_a", "big_b"), ("big_b", "big_c"), ("big_c", "big_a")],
        )
        clusters = find_zombies(graph, roots={"root"})
        assert len(clusters) == 2
        # Larger cluster first
        assert len(clusters[0].symbols) >= len(clusters[1].symbols)

    def test_alive_predecessor_blocks_cluster(self):
        # alive → semi_dead: semi_dead has an alive predecessor
        # so it shouldn't be in a zombie cluster
        graph = _make_graph(
            ["root", "alive", "semi_dead"],
            [("root", "alive"), ("alive", "semi_dead")],
        )
        clusters = find_zombies(graph, roots={"root"})
        assert len(clusters) == 0  # semi_dead is reachable from root


# ─── Deletion Type Classification ─────────────────────────────────

class TestDeletionType:
    def test_sole_symbol_in_file_gets_pure_deletion(self):
        graph = CodeGraph()
        graph.add_node(GraphNode(
            fqn="orphan.func", file_path="orphan.py",
            start_line=1, end_line=5, kind="function", name="func",
        ))
        graph.add_node(GraphNode(
            fqn="root.main", file_path="root.py",
            start_line=1, end_line=5, kind="function", name="main",
        ))
        clusters = find_zombies(graph, roots={"root.main"})
        assert len(clusters) == 1
        assert clusters[0].symbols[0].deletion_type.value == "pure_deletion"
