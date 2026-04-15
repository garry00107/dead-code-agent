"""
Layer 1 — Static Analysis: Graph-based dead code detection.

This replaces the original stub. It performs the deterministic core of the
dead code agent:

  1. Detect live roots (Tiers 1, 2, 2.5, 3)
  2. Build a directed call/import graph via AST parsing
  3. Run Tarjan's SCC to find zombie clusters
  4. Collect tool-sourced evidence for each candidate symbol
  5. Output clusters with evidence (no confidence scores yet — that's Layer 2's job)

Usage:
    python layer1_static.py \\
        --target src \\
        --output /tmp/static_candidates.json

Zero LLM involvement. 100% deterministic. Reproducible.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from graph_builder import GraphBuilder, CodeGraph
from graph_traversal import find_zombies
from models import ZombieCluster, DeadSymbol
from root_detector import RootDetector
from scoring import (
    evidence_zero_references,
    evidence_scc_isolated,
    evidence_git_stale,
    evidence_no_test_coverage,
    evidence_import_orphan,
)

logger = logging.getLogger(__name__)


class StaticAnalyzer:
    """
    Full Layer 1 pipeline: roots → graph → SCC → evidence collection.
    """

    def __init__(self, repo_path: Path, target_path: str = ""):
        self.repo_path = repo_path
        self.target_path = target_path

    def analyze(self) -> list[ZombieCluster]:
        """Run the full static analysis pipeline."""

        # ── Step 1: Detect all live roots ──────────────────────────
        logger.info("=== Step 1: Root Detection ===")
        detector = RootDetector(self.repo_path)
        roots_by_tier = detector.find_all_roots()
        all_roots = set().union(*roots_by_tier.values())
        logger.info("Total roots: %d", len(all_roots))

        # ── Step 2: Build code graph ───────────────────────────────
        logger.info("=== Step 2: Graph Construction ===")
        builder = GraphBuilder(self.repo_path, self.target_path or None)
        graph = builder.build()
        logger.info("Graph: %d nodes, %d edges", graph.node_count, graph.edge_count)

        if graph.node_count == 0:
            logger.warning("Empty graph — no source files found in target path")
            return []

        # ── Step 3: Find zombie clusters via SCC ───────────────────
        logger.info("=== Step 3: SCC Traversal ===")
        clusters = find_zombies(graph, all_roots)
        logger.info("Found %d zombie clusters", len(clusters))

        if not clusters:
            return []

        # ── Step 4: Collect evidence for each symbol ───────────────
        logger.info("=== Step 4: Evidence Collection ===")
        for cluster in clusters:
            self._collect_evidence(cluster, graph)

        return clusters

    # ─────────────────────────────────────────────
    # Evidence Collection
    # ─────────────────────────────────────────────

    def _collect_evidence(self, cluster: ZombieCluster, graph: CodeGraph) -> None:
        """Gather tool-sourced evidence for each symbol in the cluster."""
        for symbol in cluster.symbols:
            evidence = []

            # 1. Reference count (ripgrep)
            ref_evidence = self._grep_references(symbol)
            evidence.append(ref_evidence)

            # 2. SCC isolation
            scc_evidence = evidence_scc_isolated(
                scc_size=len(cluster.symbols),
                inbound_from_alive=0,  # Already filtered by traversal
            )
            evidence.append(scc_evidence)

            # 3. Git staleness
            git_evidence = self._git_staleness(symbol)
            if git_evidence:
                evidence.append(git_evidence)

            # 4. Test coverage
            test_evidence = self._check_test_references(symbol)
            evidence.append(test_evidence)

            # 5. Import orphan status
            import_evidence = self._check_import_references(symbol, graph)
            evidence.append(import_evidence)

            symbol.evidence = evidence

    def _grep_references(self, symbol: DeadSymbol) -> dict:
        """Use ripgrep or grep to count references to the symbol name."""
        simple_name = symbol.fqn.split(".")[-1]

        # Skip common names that would have too many false positives
        if simple_name in {"__init__", "__str__", "__repr__", "setUp", "tearDown",
                           "self", "cls", "args", "kwargs", "main"}:
            return evidence_zero_references(grep_count=1, total_files=0)

        try:
            # Try ripgrep first (faster)
            result = subprocess.run(
                ["rg", "--count-matches", "--no-filename", "-l", simple_name, str(self.repo_path)],
                capture_output=True, text=True, timeout=30,
                cwd=self.repo_path,
            )
            file_count = len(result.stdout.strip().splitlines()) if result.stdout.strip() else 0

            # Subtract 1 for the definition itself
            ref_count = max(0, file_count - 1)

        except (FileNotFoundError, subprocess.TimeoutExpired):
            try:
                # Fallback to grep
                result = subprocess.run(
                    ["grep", "-rl", simple_name, str(self.repo_path),
                     "--include=*.py", "--include=*.ts", "--include=*.tsx",
                     "--include=*.js", "--include=*.jsx"],
                    capture_output=True, text=True, timeout=30,
                )
                file_count = len(result.stdout.strip().splitlines()) if result.stdout.strip() else 0
                ref_count = max(0, file_count - 1)
            except (FileNotFoundError, subprocess.TimeoutExpired):
                ref_count = 0
                file_count = 0

        total_files = self._count_source_files()
        return evidence_zero_references(grep_count=ref_count, total_files=total_files)

    def _git_staleness(self, symbol: DeadSymbol) -> dict | None:
        """Check when the symbol was last modified via git blame."""
        try:
            result = subprocess.run(
                ["git", "log", "-1", "--format=%aI",
                 f"-L{symbol.start_line},{symbol.end_line}:{symbol.file_path}"],
                capture_output=True, text=True, timeout=15,
                cwd=self.repo_path,
            )
            if result.returncode != 0 or not result.stdout.strip():
                return None

            date_str = result.stdout.strip().splitlines()[0]
            try:
                last_modified = datetime.fromisoformat(date_str)
                now = datetime.now(timezone.utc)
                days_stale = (now - last_modified).days
                return evidence_git_stale(
                    last_modified=date_str[:10],
                    days_stale=days_stale,
                )
            except (ValueError, TypeError):
                return None
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None

    def _check_test_references(self, symbol: DeadSymbol) -> dict:
        """Check if any test file references this symbol."""
        simple_name = symbol.fqn.split(".")[-1]
        test_dirs = ["tests", "test", "spec", "__tests__"]
        test_files_checked = 0
        found = False

        for test_dir in test_dirs:
            test_path = self.repo_path / test_dir
            if not test_path.exists():
                continue
            for ext in ["*.py", "*.ts", "*.tsx", "*.js"]:
                for f in test_path.rglob(ext):
                    test_files_checked += 1
                    try:
                        content = f.read_text(errors="replace")
                        if simple_name in content:
                            found = True
                            break
                    except OSError:
                        continue
                if found:
                    break
            if found:
                break

        return evidence_no_test_coverage(has_tests=found, test_files_checked=test_files_checked)

    def _check_import_references(self, symbol: DeadSymbol, graph: CodeGraph) -> dict:
        """Check how many other modules import this symbol."""
        fqn = symbol.fqn
        import_count = 0

        # Count inbound edges from other files
        for source_fqn, targets in graph.edges.items():
            if fqn in targets:
                source_node = graph.nodes.get(source_fqn)
                if source_node and source_node.file_path != symbol.file_path:
                    import_count += 1

        return evidence_import_orphan(import_count=import_count)

    def _count_source_files(self) -> int:
        """Count total source files in the repo."""
        count = 0
        exclude = {".venv", "venv", "node_modules", ".git", "__pycache__", "dist"}
        for ext in ["**/*.py", "**/*.ts", "**/*.tsx", "**/*.js", "**/*.jsx"]:
            for f in self.repo_path.glob(ext):
                if not any(part in exclude for part in f.parts):
                    count += 1
        return count


# ─────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Layer 1: Static analysis for dead code detection",
    )
    parser.add_argument(
        "--target", default="",
        help="Subdirectory to scope analysis (default: entire repo)",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output path for candidates JSON",
    )
    args = parser.parse_args(argv)

    repo_path = Path.cwd()
    analyzer = StaticAnalyzer(repo_path, target_path=args.target)
    clusters = analyzer.analyze()

    # Serialize
    output_data = [c.to_dict() for c in clusters]
    Path(args.output).write_text(json.dumps(output_data, indent=2))
    logger.info("Wrote %d clusters to %s", len(clusters), args.output)


if __name__ == "__main__":
    main()
