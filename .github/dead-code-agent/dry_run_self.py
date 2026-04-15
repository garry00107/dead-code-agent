#!/usr/bin/env python3
"""
Inaugural Dry Run — point the dead code agent at ITSELF.

This is the highest-signal test: we know every function's liveness,
so every false positive teaches us exactly what the detectors missed
and every true positive validates the pipeline end-to-end.

Usage:
    python .github/dead-code-agent/dry_run_self.py

Output:
    - Shadow report JSON in reports/
    - Markdown summary to stdout
    - Console log of every decision with reasoning
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Fixup path for direct invocation
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from graph_builder import GraphBuilder
from graph_traversal import find_zombies
from root_detector import RootDetector
from scoring import (
    calculate_confidence,
    calculate_cluster_confidence,
    evidence_zero_references,
    evidence_scc_isolated,
    evidence_import_orphan,
    evidence_no_test_coverage,
)
from shadow_report import (
    build_shadow_report,
    save_report,
    report_to_markdown,
    generate_run_id,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("dry_run_self")


def main() -> None:
    repo_root = _SCRIPT_DIR.parent.parent  # .github/dead-code-agent → repo root
    target_dir = ".github/dead-code-agent"

    logger.info("═" * 60)
    logger.info("INAUGURAL DRY RUN — Agent Self-Analysis")
    logger.info("═" * 60)
    logger.info("Repo root:  %s", repo_root)
    logger.info("Target:     %s", target_dir)

    # ── Phase 1: Root Detection ──────────────────────────────────
    logger.info("")
    logger.info("Phase 1: Root Detection")
    logger.info("─" * 40)

    detector = RootDetector(repo_root)
    roots_by_tier = detector.find_all_roots()
    all_roots = set().union(*roots_by_tier.values())

    total_roots = len(all_roots)
    logger.info("Total unique roots: %d", total_roots)
    for tier, roots in roots_by_tier.items():
        if roots:
            logger.info("  %s: %d roots", tier, len(roots))
            for r in sorted(roots)[:5]:
                logger.info("    └─ %s", r)
            if len(roots) > 5:
                logger.info("    └─ ... and %d more", len(roots) - 5)

    # ── Phase 2: Graph Construction ──────────────────────────────
    logger.info("")
    logger.info("Phase 2: Graph Construction")
    logger.info("─" * 40)

    builder = GraphBuilder(repo_root, target_dir)
    graph = builder.build()

    logger.info("Nodes: %d", graph.node_count)
    logger.info("Edges: %d", graph.edge_count)
    logger.info("Files: %d", len(graph.file_symbols))

    # Show files and their symbol counts
    for filepath, symbols in sorted(graph.file_symbols.items()):
        logger.info("  📄 %s: %d symbols", filepath, len(symbols))

    # ── Phase 3: Zombie Detection ────────────────────────────────
    logger.info("")
    logger.info("Phase 3: Zombie Detection (Tarjan's SCC)")
    logger.info("─" * 40)

    clusters = find_zombies(graph, roots=all_roots)
    logger.info("Zombie clusters found: %d", len(clusters))

    # ── Phase 4: Evidence & Scoring ──────────────────────────────
    logger.info("")
    logger.info("Phase 4: Evidence Collection & Scoring")
    logger.info("─" * 40)

    scored_clusters = []
    for cluster in clusters:
        symbol_scores = []
        for sym in cluster.symbols:
            # Gather deterministic evidence (no LLM in self-test)
            evidence = []

            # Reference counting via graph edges
            inbound = sum(
                1 for edges in graph.edges.values()
                for target in edges if target == sym.fqn
            )
            evidence.append(evidence_zero_references(inbound, graph.node_count))

            # SCC isolation
            alive_inbound = sum(
                1 for src, edges in graph.edges.items()
                if sym.fqn in edges and src in all_roots
            )
            evidence.append(evidence_scc_isolated(
                scc_size=len(cluster.symbols),
                inbound_from_alive=alive_inbound,
            ))

            # Import orphan check
            is_imported = any(
                sym.fqn in edges
                for edges in graph.edges.values()
            )
            evidence.append(evidence_import_orphan(
                import_count=1 if is_imported else 0,
            ))

            # Test coverage check
            sym_name = sym.fqn.split(".")[-1]
            has_tests = any(
                sym_name in str(graph.file_symbols.get(f, []))
                for f in graph.file_symbols
                if "test" in f.lower()
            )
            evidence.append(evidence_no_test_coverage(has_tests, test_files_checked=11))

            score = calculate_confidence(evidence)
            sym.confidence = score
            sym.evidence = evidence
            symbol_scores.append(score)

        cluster.combined_confidence = calculate_cluster_confidence(symbol_scores, strategy="min")
        scored_clusters.append(cluster)

    # ── Results ──────────────────────────────────────────────────
    threshold = 0.92
    passing = [c for c in scored_clusters if c.combined_confidence >= threshold]
    below = [c for c in scored_clusters if c.combined_confidence < threshold]

    logger.info("")
    logger.info("═" * 60)
    logger.info("RESULTS")
    logger.info("═" * 60)
    logger.info("Total clusters:        %d", len(scored_clusters))
    logger.info("Above threshold (≥92%%): %d", len(passing))
    logger.info("Below threshold:       %d", len(below))

    if passing:
        logger.info("")
        logger.info("🔴 WOULD OPEN PRs FOR:")
        for cluster in passing:
            logger.info("  Cluster %s (confidence: %.1f%%)",
                        cluster.cluster_id[:8], cluster.combined_confidence * 100)
            for sym in cluster.symbols:
                logger.info("    ├─ %s (%s:%d-%d) — %.1f%%",
                            sym.fqn, sym.file_path, sym.start_line, sym.end_line,
                            sym.confidence * 100)

    if below:
        logger.info("")
        logger.info("⚪ BELOW THRESHOLD (skipped):")
        for cluster in below:
            logger.info("  Cluster %s (confidence: %.1f%%)",
                        cluster.cluster_id[:8], cluster.combined_confidence * 100)
            for sym in cluster.symbols[:5]:
                logger.info("    ├─ %s — %.1f%%", sym.fqn, sym.confidence * 100)
            if len(cluster.symbols) > 5:
                logger.info("    └─ ... and %d more symbols", len(cluster.symbols) - 5)

    if not scored_clusters:
        logger.info("")
        logger.info("✅ No dead code found. All symbols are reachable from detected roots.")
        logger.info("   This is the expected outcome for a freshly-built codebase.")

    # ── Shadow Report ────────────────────────────────────────────
    logger.info("")
    logger.info("Phase 5: Shadow Report Generation")
    logger.info("─" * 40)

    run_id = generate_run_id()
    report = build_shadow_report(
        run_id=run_id,
        all_clusters=scored_clusters,
        threshold_passing=passing,
        roots_by_tier={k: v for k, v in roots_by_tier.items()},
        proposed_patterns=[],
        config={
            "threshold": threshold,
            "target": target_dir,
            "mode": "self_test",
        },
        run_url="local://self-test",
    )

    reports_dir = _SCRIPT_DIR / "reports"
    report_path = save_report(report, reports_dir)
    logger.info("Report saved: %s", report_path)

    # Print markdown summary
    md = report_to_markdown(report)
    logger.info("")
    logger.info("═" * 60)
    print(md)

    # ── Verdict ──────────────────────────────────────────────────
    logger.info("")
    logger.info("═" * 60)
    if len(passing) == 0 and len(scored_clusters) == 0:
        logger.info("🎯 PASS — Zero clusters, zero dead code (clean codebase)")
    elif len(passing) == 0:
        logger.info("🎯 PASS — %d clusters found but all below threshold", len(scored_clusters))
    else:
        logger.info("⚠️  REVIEW — %d clusters would trigger PRs. Triage required.", len(passing))
    logger.info("═" * 60)


if __name__ == "__main__":
    main()
