"""
PR Generator — CLI entry point called by the GitHub Actions workflow.

Usage:
    python pr_generator.py \\
        --clusters /tmp/clusters.json \\
        --batch-index 0 \\
        --repo "owner/repo" \\
        --base-branch main \\
        --output /tmp/pr_result_0.json

Exit codes:
    0 — always (even on safety-gate blocks — those are expected outcomes)
    1 — only for unrecoverable infrastructure errors (can't read files, etc.)

This is the orchestrator. The flow is:
    Load cluster → Safety Gate → Plan → Branch → Mutate → Smoke Test → Commit → PR → Result
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Support both direct invocation and package import
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from models import ZombieCluster, PRResult, load_clusters_from_json, save_pr_result
from safety_gate import SafetyGate
from deletion_planner import DeletionPlanner
from git_operations import GitOperations
from pr_description import PRDescriptionGenerator
from smoke_test import SmokeTestRunner

logger = logging.getLogger(__name__)


class PRGeneratorOrchestrator:
    """
    Main orchestrator that converts a confirmed dead-code cluster into a PR.
    Each call handles exactly ONE cluster (one PR = one cluster = atomic).
    """

    def __init__(
        self,
        repo_path: Path,
        github_token: str,
        repo_slug: str,
        base_branch: str = "main",
        confidence_threshold: float = 0.92,
    ):
        self.repo_path = repo_path
        self.base_branch = base_branch

        self.safety_gate = SafetyGate(
            confidence_threshold=confidence_threshold,
            github_token=github_token,
            repo_slug=repo_slug,
        )
        self.planner = DeletionPlanner(repo_path=repo_path)
        self.describer = PRDescriptionGenerator()
        self.git = GitOperations(
            repo_path=repo_path,
            github_token=github_token,
            repo_slug=repo_slug,
        )
        self.smoke_tester = SmokeTestRunner(repo_path=repo_path)

    def generate_pr(self, cluster: ZombieCluster) -> PRResult:
        """
        Full pipeline: safety → plan → branch → mutate → test → commit → PR.
        Never raises — returns PRResult with success=False on any failure.
        """
        branch_name = f"dead-code/cluster-{cluster.cluster_id[:8]}"

        try:
            # ── 1. Safety gate ──────────────────────────────────────
            logger.info(
                "=== Processing cluster %s (%d symbols, %.0f%% confidence) ===",
                cluster.cluster_id[:8],
                len(cluster.symbols),
                cluster.combined_confidence * 100,
            )

            ok, failures = self.safety_gate.run_all(cluster, self.repo_path)
            if not ok:
                logger.warning(
                    "Safety gate blocked cluster %s: %s",
                    cluster.cluster_id[:8],
                    "; ".join(failures),
                )
                return PRResult(
                    success=False,
                    pr_url=None,
                    branch_name=branch_name,
                    symbols_deleted=[],
                    files_modified=[],
                    confidence=cluster.combined_confidence,
                    error=f"Safety gate blocked: {'; '.join(failures)}",
                    requires_human_review=True,
                )

            # ── 2. Plan deletions ───────────────────────────────────
            steps = self.planner.plan(cluster)
            logger.info("Planned %d deletion steps", len(steps))

            # Validate plan before executing
            valid, plan_errors = self.planner.validate_plan(steps, self.repo_path)
            if not valid:
                return PRResult(
                    success=False,
                    pr_url=None,
                    branch_name=branch_name,
                    symbols_deleted=[],
                    files_modified=[],
                    confidence=cluster.combined_confidence,
                    error=f"Plan validation failed: {'; '.join(plan_errors)}",
                    requires_human_review=True,
                )

            # ── 3. Create branch ────────────────────────────────────
            branch_name = self.git.create_branch(cluster.cluster_id)
            logger.info("Created branch: %s", branch_name)

            # ── 4. Apply file mutations ─────────────────────────────
            self.git.apply_deletion_plan(steps)

            # ── 5. Smoke test ───────────────────────────────────────
            modified_files = [s["file"] for s in steps if s["action"] != "cleanup_imports"]
            deleted_files = [s["file"] for s in steps if s["action"] == "delete_file"]

            smoke_result = self.smoke_tester.run(modified_files, deleted_files)

            if not smoke_result.passed:
                logger.warning(
                    "Smoke tests failed for cluster %s — aborting branch\n%s",
                    cluster.cluster_id[:8],
                    smoke_result.details,
                )
                self.git.abort_branch(branch_name)
                return PRResult(
                    success=False,
                    pr_url=None,
                    branch_name=branch_name,
                    symbols_deleted=[],
                    files_modified=[],
                    confidence=cluster.combined_confidence,
                    error=f"Smoke tests failed after deletion — branch abandoned\n{smoke_result.details}",
                    requires_human_review=True,
                )

            logger.info("Smoke tests passed:\n%s", smoke_result.details)

            # ── 6. Commit & push ────────────────────────────────────
            sha = self.git.commit_all(cluster, branch_name)
            logger.info("Committed: %s", sha[:8])

            # ── 7. Generate PR description ──────────────────────────
            pr_meta = self.describer.generate(cluster, steps)

            # ── 8. Open PR ──────────────────────────────────────────
            pr_url = self.git.open_pull_request(
                branch=branch_name,
                pr_meta=pr_meta,
                base_branch=self.base_branch,
                labels=["chore"],
            )
            logger.info("Opened PR: %s", pr_url)

            return PRResult(
                success=True,
                pr_url=pr_url,
                branch_name=branch_name,
                symbols_deleted=[s.fqn for s in cluster.symbols],
                files_modified=list(set(s.file_path for s in cluster.symbols)),
                confidence=cluster.combined_confidence,
                title=pr_meta["title"],
            )

        except Exception as exc:
            logger.exception("Unexpected error processing cluster %s", cluster.cluster_id[:8])
            self.git.abort_branch(branch_name)
            return PRResult(
                success=False,
                pr_url=None,
                branch_name=branch_name,
                symbols_deleted=[],
                files_modified=[],
                confidence=cluster.combined_confidence,
                error=str(exc),
                requires_human_review=True,
            )


# ─────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a dead-code removal PR for a single cluster.",
    )
    parser.add_argument(
        "--clusters", required=True,
        help="Path to clusters JSON file from analysis phase",
    )
    parser.add_argument(
        "--batch-index", type=int, required=True,
        help="Index of the cluster to process (0-based)",
    )
    parser.add_argument(
        "--repo", required=True,
        help="GitHub repository slug (owner/repo)",
    )
    parser.add_argument(
        "--base-branch", default="main",
        help="Base branch for the PR (default: main)",
    )
    parser.add_argument(
        "--output", required=True,
        help="Path to write the PR result JSON",
    )
    parser.add_argument(
        "--confidence-threshold", type=float, default=0.92,
        help="Minimum confidence threshold (default: 0.92)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """
    CLI main. Returns 0 always (even on safety blocks).
    Returns 1 only for infrastructure errors.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = parse_args(argv)

    # Load clusters
    try:
        clusters = load_clusters_from_json(args.clusters)
    except Exception as exc:
        logger.error("Failed to load clusters from %s: %s", args.clusters, exc)
        return 1

    # Check if batch index is valid
    if args.batch_index >= len(clusters):
        logger.info(
            "Batch index %d exceeds cluster count %d — nothing to do",
            args.batch_index,
            len(clusters),
        )
        # Write a "skipped" result so downstream jobs don't break
        result = PRResult(
            success=False,
            pr_url=None,
            branch_name="",
            symbols_deleted=[],
            files_modified=[],
            error="Batch index exceeds cluster count — skipped",
        )
        save_pr_result(result, args.output)
        return 0

    cluster = clusters[args.batch_index]

    # Get GitHub token
    github_token = os.environ.get("GITHUB_TOKEN", "")
    if not github_token:
        logger.error("GITHUB_TOKEN environment variable not set")
        return 1

    # Run the orchestrator
    repo_path = Path.cwd()
    orchestrator = PRGeneratorOrchestrator(
        repo_path=repo_path,
        github_token=github_token,
        repo_slug=args.repo,
        base_branch=args.base_branch,
        confidence_threshold=args.confidence_threshold,
    )

    result = orchestrator.generate_pr(cluster)

    # Write result
    save_pr_result(result, args.output)
    logger.info(
        "Result written to %s — success=%s, pr_url=%s",
        args.output,
        result.success,
        result.pr_url,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
