"""
Layer 2 — Agent Investigation: LLM adversarial review + confidence scoring.

Takes the static candidates from Layer 1 and:
  1. Reads the actual code for each symbol
  2. Runs the adversarial LLM judge (defense attorney prompt)
  3. Calculates deterministic confidence scores
  4. Collects pattern proposals for learned_patterns.yml
  5. Filters by confidence threshold

The LLM is NEVER the final arbiter. It adds weak evidence signals that
the mathematical scoring formula incorporates. Tools discover, LLM challenges,
math scores.

Usage:
    python layer2_agent.py \\
        --candidates /tmp/static_candidates.json \\
        --threshold 0.92 \\
        --output /tmp/clusters.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from adversarial_judge import AdversarialJudge
from graph_builder import GraphNode
from models import ZombieCluster, DeadSymbol, load_clusters_from_json
from scoring import (
    calculate_confidence,
    calculate_cluster_confidence,
    ScoringConfig,
)

logger = logging.getLogger(__name__)


class AgentInvestigator:
    """
    Layer 2 pipeline: LLM review + confidence scoring + pattern collection.
    """

    def __init__(
        self,
        repo_path: Path,
        confidence_threshold: float = 0.92,
        api_key: Optional[str] = None,
        skip_llm: bool = False,
    ):
        self.repo_path = repo_path
        self.threshold = confidence_threshold
        self.skip_llm = skip_llm
        self.judge = AdversarialJudge(api_key=api_key) if not skip_llm else None
        self.scoring_config = ScoringConfig()
        self.proposed_patterns: list[dict] = []

    def investigate(self, clusters: list[ZombieCluster]) -> list[ZombieCluster]:
        """
        Run Layer 2 investigation on all clusters.

        Returns only clusters that meet the confidence threshold.
        """
        logger.info("=== Layer 2: Agent Investigation ===")
        logger.info("Clusters to investigate: %d", len(clusters))
        logger.info("Confidence threshold: %.0f%%", self.threshold * 100)
        logger.info("LLM review: %s", "enabled" if not self.skip_llm else "disabled")

        scored_clusters: list[ZombieCluster] = []

        for i, cluster in enumerate(clusters):
            logger.info(
                "--- Cluster %d/%d: %s (%d symbols) ---",
                i + 1, len(clusters), cluster.cluster_id[:8], len(cluster.symbols),
            )

            # Step 1: LLM adversarial review (if enabled)
            if self.judge:
                self._run_llm_review(cluster)

            # Step 2: Calculate confidence scores
            self._score_cluster(cluster)

            logger.info(
                "Cluster %s confidence: %.0f%% (threshold: %.0f%%)",
                cluster.cluster_id[:8],
                cluster.combined_confidence * 100,
                self.threshold * 100,
            )

            # Step 3: Filter by threshold
            if cluster.combined_confidence >= self.threshold:
                scored_clusters.append(cluster)
                logger.info("✓ Above threshold — queued for PR")
            else:
                logger.info("✗ Below threshold — skipped")

        logger.info(
            "=== Layer 2 complete: %d/%d clusters above threshold ===",
            len(scored_clusters), len(clusters),
        )

        # Log pattern proposals
        if self.proposed_patterns:
            logger.info(
                "LLM proposed %d new patterns for learned_patterns.yml",
                len(self.proposed_patterns),
            )

        return scored_clusters

    # ─────────────────────────────────────────────
    # LLM Review
    # ─────────────────────────────────────────────

    def _run_llm_review(self, cluster: ZombieCluster) -> None:
        """Run the adversarial judge on each symbol in the cluster."""
        for symbol in cluster.symbols:
            # Read the actual code
            code_snippet = self._read_code(symbol)

            # Build a GraphNode for the judge
            node = GraphNode(
                fqn=symbol.fqn,
                file_path=symbol.file_path,
                start_line=symbol.start_line,
                end_line=symbol.end_line,
                kind="function",  # Simplified — judge doesn't need exact kind
                name=symbol.fqn.split(".")[-1],
            )

            # Run adversarial assessment
            llm_evidence = self.judge.assess_symbol(
                node=node,
                code_snippet=code_snippet,
                tool_evidence=symbol.evidence,
            )

            # Append LLM evidence to existing tool evidence
            symbol.evidence.append(llm_evidence)

            # Collect pattern proposals
            if "_proposed_pattern" in llm_evidence:
                pattern = llm_evidence.pop("_proposed_pattern")
                pattern["discovered"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                self.proposed_patterns.append(pattern)
                logger.info(
                    "LLM proposed pattern: type=%s, match=%s (from %s)",
                    pattern.get("type"), pattern.get("match"), symbol.fqn,
                )

    # ─────────────────────────────────────────────
    # Scoring
    # ─────────────────────────────────────────────

    def _score_cluster(self, cluster: ZombieCluster) -> None:
        """Calculate confidence scores for all symbols and the cluster."""
        symbol_scores = []

        for symbol in cluster.symbols:
            score = calculate_confidence(symbol.evidence, self.scoring_config)
            symbol.confidence = score
            symbol_scores.append(score)
            logger.debug("  %s: %.0f%%", symbol.fqn, score * 100)

        # Cluster confidence = minimum of all symbol scores (most conservative)
        cluster.combined_confidence = calculate_cluster_confidence(
            symbol_scores, strategy="min",
        )

    # ─────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────

    def _read_code(self, symbol: DeadSymbol) -> str:
        """Read the source code for a symbol from the repo."""
        file_path = self.repo_path / symbol.file_path
        if not file_path.exists():
            return "# File not found"

        try:
            lines = file_path.read_text(errors="replace").splitlines()
            start = max(0, symbol.start_line - 1)
            end = min(len(lines), symbol.end_line)
            return "\n".join(lines[start:end])
        except OSError:
            return "# Error reading file"

    def write_pattern_proposals(self, output_path: Optional[Path] = None) -> None:
        """Write any LLM-proposed patterns to a JSON file for later review."""
        if not self.proposed_patterns:
            return

        path = output_path or (self.repo_path / ".github" / "dead-code-agent" / "proposed_patterns.json")
        path.parent.mkdir(parents=True, exist_ok=True)

        # Merge with existing proposals if file exists
        existing = []
        if path.exists():
            try:
                existing = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                pass

        # Deduplicate by (type, match)
        seen = {(p["type"], p["match"]) for p in existing}
        for p in self.proposed_patterns:
            key = (p["type"], p["match"])
            if key not in seen:
                existing.append(p)
                seen.add(key)

        path.write_text(json.dumps(existing, indent=2))
        logger.info("Wrote %d pattern proposals to %s", len(existing), path)


# ─────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Layer 2: Agent investigation with LLM adversarial review",
    )
    parser.add_argument(
        "--candidates", required=True,
        help="Path to static candidates JSON from Layer 1",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.92,
        help="Minimum confidence threshold (default: 0.92)",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output path for scored clusters JSON",
    )
    parser.add_argument(
        "--skip-llm", action="store_true",
        help="Skip LLM review (score from tool evidence only)",
    )
    args = parser.parse_args(argv)

    repo_path = Path.cwd()
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    # Load candidates from Layer 1
    clusters = load_clusters_from_json(args.candidates)
    logger.info("Loaded %d candidate clusters from %s", len(clusters), args.candidates)

    # Run investigation
    investigator = AgentInvestigator(
        repo_path=repo_path,
        confidence_threshold=args.threshold,
        api_key=api_key,
        skip_llm=args.skip_llm or not api_key,
    )
    scored = investigator.investigate(clusters)

    # Write scored clusters
    output_data = [c.to_dict() for c in scored]
    Path(args.output).write_text(json.dumps(output_data, indent=2))
    logger.info("Wrote %d threshold-passing clusters to %s", len(scored), args.output)

    # Write pattern proposals
    investigator.write_pattern_proposals()


if __name__ == "__main__":
    main()
