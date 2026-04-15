"""
Graph Traversal — Tarjan's SCC algorithm + zombie cluster identification.

This is the core deterministic algorithm that finds dead code.

How it works:
  1. Build a directed graph (done by graph_builder.py)
  2. Identify roots (done by root_detector.py)
  3. Forward-propagate "alive" status from roots through edges
  4. Group remaining (unreachable) nodes into SCCs via Tarjan's algorithm
  5. Each SCC with zero inbound edges from alive nodes = a zombie cluster

The SCC grouping is critical because it handles mutual recursion:
  - A calls B, B calls A, but nothing else calls either → both are dead
  - Without SCC, you'd think A is alive (because B calls it) and vice versa

Zero LLM involvement. Pure graph theory.
"""

from __future__ import annotations

import hashlib
import logging
import sys
from pathlib import Path
from typing import Optional

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from graph_builder import CodeGraph, GraphNode
from models import DeadSymbol, DeletionType, ZombieCluster

logger = logging.getLogger(__name__)


class GraphTraversal:
    """
    Identifies zombie clusters using SCC decomposition and root reachability.
    """

    def __init__(self, graph: CodeGraph, roots: set[str]):
        self.graph = graph
        self.roots = roots

    def find_zombie_clusters(self) -> list[ZombieCluster]:
        """
        Main entry point. Returns a list of ZombieClusters — groups of
        dead symbols that are only reachable from each other.

        Steps:
          1. Mark all root-reachable nodes as alive
          2. Compute SCCs on the remaining (dead) subgraph
          3. Filter SCCs to only those with zero inbound edges from alive nodes
          4. Package each SCC as a ZombieCluster
        """
        # Step 1: Forward-propagate "alive" from roots
        alive = self._compute_alive_set()
        logger.info("Alive nodes: %d / %d total", len(alive), self.graph.node_count)

        # Dead nodes = everything not alive
        dead_fqns = set(self.graph.nodes.keys()) - alive
        logger.info("Dead candidates: %d", len(dead_fqns))

        if not dead_fqns:
            return []

        # Step 2: Compute SCCs on the dead subgraph
        sccs = self._tarjan_scc(dead_fqns)
        logger.info("Found %d SCCs among dead nodes", len(sccs))

        # Step 3: Filter — only keep SCCs with NO inbound edges from alive nodes
        # (This filters out nodes that look dead but might be reachable
        #  through paths we didn't trace — conservative safety check)
        zombie_sccs = []
        for scc in sccs:
            if not self._has_alive_predecessor(scc, alive):
                zombie_sccs.append(scc)

        logger.info("Zombie clusters (no alive predecessors): %d", len(zombie_sccs))

        # Step 4: Package as ZombieClusters
        clusters = []
        for scc in zombie_sccs:
            cluster = self._scc_to_zombie_cluster(scc)
            if cluster:
                clusters.append(cluster)

        # Sort by cluster size (largest first — most impactful)
        clusters.sort(key=lambda c: len(c.symbols), reverse=True)
        return clusters

    # ─────────────────────────────────────────────
    # Step 1: Alive Set Computation
    # ─────────────────────────────────────────────

    def _compute_alive_set(self) -> set[str]:
        """
        BFS/DFS from all roots, following edges to mark nodes as alive.
        Any node reachable from a root is alive.

        Also handles glob patterns in roots (from Tier 3 declarations).
        """
        import fnmatch

        alive = set()
        queue: list[str] = []

        # Seed with all root nodes
        for fqn in self.graph.nodes:
            # Exact match
            if fqn in self.roots:
                alive.add(fqn)
                queue.append(fqn)
                continue

            # Glob match
            for pattern in self.roots:
                if "*" in pattern and fnmatch.fnmatch(fqn, pattern):
                    alive.add(fqn)
                    queue.append(fqn)
                    break

        logger.info("Root seeds: %d nodes", len(queue))

        # BFS: follow all edges from alive nodes
        while queue:
            current = queue.pop()
            for neighbor in self.graph.edges.get(current, set()):
                if neighbor not in alive:
                    alive.add(neighbor)
                    queue.append(neighbor)

        return alive

    # ─────────────────────────────────────────────
    # Step 2: Tarjan's SCC Algorithm
    # ─────────────────────────────────────────────

    def _tarjan_scc(self, node_set: set[str]) -> list[list[str]]:
        """
        Tarjan's Strongly Connected Components algorithm.

        Returns a list of SCCs, each SCC being a list of FQNs.
        Only considers nodes in node_set and edges between them.

        Implementation uses an iterative approach to avoid stack overflow
        on large codebases.
        """
        index_counter = [0]
        stack: list[str] = []
        on_stack: set[str] = set()
        index: dict[str, int] = {}
        lowlink: dict[str, int] = {}
        sccs: list[list[str]] = []

        def strongconnect(v: str) -> None:
            # Use an explicit work stack to avoid Python recursion limits
            work_stack: list[tuple[str, list[str], int]] = []
            index[v] = index_counter[0]
            lowlink[v] = index_counter[0]
            index_counter[0] += 1
            stack.append(v)
            on_stack.add(v)

            # Get neighbors within the dead subgraph
            neighbors = [
                n for n in self.graph.edges.get(v, set())
                if n in node_set
            ]
            work_stack.append((v, neighbors, 0))

            while work_stack:
                node, nbrs, i = work_stack[-1]

                if i < len(nbrs):
                    w = nbrs[i]
                    work_stack[-1] = (node, nbrs, i + 1)

                    if w not in index:
                        # w has not been visited — recurse
                        index[w] = index_counter[0]
                        lowlink[w] = index_counter[0]
                        index_counter[0] += 1
                        stack.append(w)
                        on_stack.add(w)

                        w_neighbors = [
                            n for n in self.graph.edges.get(w, set())
                            if n in node_set
                        ]
                        work_stack.append((w, w_neighbors, 0))
                    elif w in on_stack:
                        lowlink[node] = min(lowlink[node], index[w])
                else:
                    # Done processing all neighbors of node
                    if lowlink[node] == index[node]:
                        # node is root of an SCC
                        scc: list[str] = []
                        while True:
                            w = stack.pop()
                            on_stack.discard(w)
                            scc.append(w)
                            if w == node:
                                break
                        sccs.append(scc)

                    work_stack.pop()
                    if work_stack:
                        parent = work_stack[-1][0]
                        lowlink[parent] = min(lowlink[parent], lowlink[node])

        for v in node_set:
            if v not in index:
                strongconnect(v)

        return sccs

    # ─────────────────────────────────────────────
    # Step 3: Alive Predecessor Check
    # ─────────────────────────────────────────────

    def _has_alive_predecessor(self, scc: list[str], alive: set[str]) -> bool:
        """
        Check if any node in the SCC has an inbound edge from an alive node.
        If so, this SCC might not be truly dead (conservative safety check).
        """
        scc_set = set(scc)
        # Build reverse edge lookup for the SCC
        for fqn in self.graph.nodes:
            if fqn in alive:
                for target in self.graph.edges.get(fqn, set()):
                    if target in scc_set:
                        return True
        return False

    # ─────────────────────────────────────────────
    # Step 4: Cluster Packaging
    # ─────────────────────────────────────────────

    def _scc_to_zombie_cluster(self, scc: list[str]) -> Optional[ZombieCluster]:
        """Convert an SCC into a ZombieCluster with DeadSymbol entries."""
        if not scc:
            return None

        symbols: list[DeadSymbol] = []
        files_affected: set[str] = set()

        for fqn in scc:
            node = self.graph.nodes.get(fqn)
            if not node:
                continue

            files_affected.add(node.file_path)

            # Determine deletion type
            file_fqns = self.graph.file_symbols.get(node.file_path, [])
            alive_in_file = any(
                f for f in file_fqns
                if f not in set(scc)  # other symbols in file that are NOT in this SCC
            )

            if alive_in_file:
                deletion_type = DeletionType.INLINE_REMOVAL
            else:
                # All symbols in this file are in the SCC — full file delete
                all_file_fqns_in_scc = all(f in set(scc) for f in file_fqns)
                deletion_type = (
                    DeletionType.PURE_DELETION if all_file_fqns_in_scc
                    else DeletionType.INLINE_REMOVAL
                )

            symbols.append(DeadSymbol(
                fqn=fqn,
                file_path=node.file_path,
                start_line=node.start_line,
                end_line=node.end_line,
                confidence=0.0,  # Will be set by scoring.py after LLM evaluation
                evidence=[],     # Will be populated by layer2_agent.py
                zombie_cluster_id="",  # Set below
                deletion_type=deletion_type,
            ))

        if not symbols:
            return None

        # Deterministic cluster ID based on sorted FQNs
        cluster_hash = hashlib.sha256(
            "|".join(sorted(s.fqn for s in symbols)).encode()
        ).hexdigest()[:16]

        for s in symbols:
            s.zombie_cluster_id = cluster_hash

        return ZombieCluster(
            cluster_id=cluster_hash,
            symbols=symbols,
            combined_confidence=0.0,  # Set by scoring.py
            files_affected=sorted(files_affected),
        )


# ─────────────────────────────────────────────
# Convenience function
# ─────────────────────────────────────────────

def find_zombies(graph: CodeGraph, roots: set[str]) -> list[ZombieCluster]:
    """One-liner API for the full traversal."""
    traversal = GraphTraversal(graph, roots)
    return traversal.find_zombie_clusters()
