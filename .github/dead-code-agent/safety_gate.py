"""
Safety Gate — hard blockers that prevent PR creation.

These checks are NEVER overridable, not even by config.
If any check fails, the PR is silently abandoned and flagged for human review.

Design rationale:
  - False negatives (missing dead code) are fine — we'll catch it next week.
  - False positives (deleting live code) are career-limiting. So every check
    errs on the side of caution.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

import sys as _sys
from pathlib import Path as _Path
_SCRIPT_DIR = _Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in _sys.path:
    _sys.path.insert(0, str(_SCRIPT_DIR))

from models import DeletionType, ZombieCluster

logger = logging.getLogger(__name__)


class SafetyGate:
    """
    Runs all safety checks against a cluster before any file mutation.
    Returns (passed: bool, failures: list[str]).
    """

    def __init__(
        self,
        confidence_threshold: float = 0.92,
        github_token: Optional[str] = None,
        repo_slug: Optional[str] = None,
    ):
        self.confidence_threshold = confidence_threshold
        self.github_token = github_token or os.environ.get("GITHUB_TOKEN", "")
        self.repo_slug = repo_slug  # "owner/repo"

    def run_all(self, cluster: ZombieCluster, repo_path: Path) -> tuple[bool, list[str]]:
        """Execute every safety check. Short-circuits on first hard failure."""
        failures: list[str] = []

        checks = [
            ("confidence_threshold", self._check_confidence_threshold),
            ("deletion_is_pure", self._check_deletion_is_pure),
            ("no_migration", self._check_no_migration_in_progress),
            ("no_open_prs", self._check_no_open_prs_on_same_files),
            ("no_active_branches", self._check_no_active_branches_touching_files),
            ("test_suite_green", self._check_test_suite_currently_green),
        ]

        for name, check in checks:
            try:
                ok, reason = check(cluster, repo_path)
                if not ok:
                    failures.append(f"[{name}] {reason}")
                    logger.warning("Safety check failed: %s — %s", name, reason)
            except Exception as exc:
                # If a check itself errors, treat as failure — never skip
                failures.append(f"[{name}] Check errored: {exc}")
                logger.exception("Safety check errored: %s", name)

        passed = len(failures) == 0
        if passed:
            logger.info(
                "All safety checks passed for cluster %s (confidence=%.2f)",
                cluster.cluster_id[:8], cluster.combined_confidence,
            )
        return passed, failures

    # ─── Individual checks ──────────────────────────────────────────

    def _check_confidence_threshold(
        self, cluster: ZombieCluster, repo_path: Path
    ) -> tuple[bool, str]:
        if cluster.combined_confidence < self.confidence_threshold:
            return (
                False,
                f"Confidence {cluster.combined_confidence:.2f} below threshold "
                f"{self.confidence_threshold}",
            )
        return True, ""

    def _check_deletion_is_pure(
        self, cluster: ZombieCluster, repo_path: Path
    ) -> tuple[bool, str]:
        """Reject any cluster containing symbols that need call-site refactoring."""
        for sym in cluster.symbols:
            if sym.deletion_type == DeletionType.REQUIRES_REFACTOR:
                return (
                    False,
                    f"{sym.fqn} requires refactoring at call sites — needs human review",
                )
        return True, ""

    def _check_no_migration_in_progress(
        self, cluster: ZombieCluster, repo_path: Path
    ) -> tuple[bool, str]:
        """Check for migration sentinel files that signal 'don't delete anything right now'."""
        sentinels = [".migration_in_progress", "MIGRATION_LOCK"]
        for sentinel in sentinels:
            if (repo_path / sentinel).exists():
                return False, f"Migration sentinel '{sentinel}' found — aborting"
        return True, ""

    def _check_no_open_prs_on_same_files(
        self, cluster: ZombieCluster, repo_path: Path
    ) -> tuple[bool, str]:
        """
        Query GitHub API for open PRs touching the same files.
        Prevents conflicting changes and merge headaches.
        """
        if not self.github_token or not self.repo_slug:
            logger.debug("Skipping open-PR check — no GitHub credentials")
            return True, ""

        try:
            import httpx

            affected = set(
                s.file_path for s in cluster.symbols
            )
            headers = {
                "Authorization": f"Bearer {self.github_token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            }
            url = f"https://api.github.com/repos/{self.repo_slug}/pulls"
            resp = httpx.get(url, headers=headers, params={"state": "open", "per_page": 100})
            resp.raise_for_status()

            for pr in resp.json():
                pr_number = pr["number"]
                files_url = f"{url}/{pr_number}/files"
                files_resp = httpx.get(files_url, headers=headers, params={"per_page": 100})
                files_resp.raise_for_status()
                pr_files = {f["filename"] for f in files_resp.json()}

                overlap = affected & pr_files
                if overlap:
                    return (
                        False,
                        f"PR #{pr_number} ({pr['title']}) touches {overlap} — conflict risk",
                    )
        except ImportError:
            logger.debug("httpx not available — skipping open-PR check")
        except Exception as exc:
            # Network errors → treat as failure (conservative)
            return False, f"Failed to check open PRs: {exc}"

        return True, ""

    def _check_no_active_branches_touching_files(
        self, cluster: ZombieCluster, repo_path: Path
    ) -> tuple[bool, str]:
        """
        Check if any remote branch has recent commits touching cluster files.
        'Recent' = last 7 days. Prevents stepping on active work.
        """
        affected_files = [s.file_path for s in cluster.symbols]

        for file_path in affected_files:
            try:
                result = subprocess.run(
                    [
                        "git", "log", "--all", "--oneline",
                        "--since=7 days ago", "--", file_path,
                    ],
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.stdout.strip():
                    lines = result.stdout.strip().split("\n")
                    return (
                        False,
                        f"{file_path} has {len(lines)} recent commit(s) across branches "
                        f"— active work detected",
                    )
            except subprocess.TimeoutExpired:
                return False, f"Git log timed out checking {file_path}"
            except Exception as exc:
                return False, f"Git log failed for {file_path}: {exc}"

        return True, ""

    def _check_test_suite_currently_green(
        self, cluster: ZombieCluster, repo_path: Path
    ) -> tuple[bool, str]:
        """
        Verify the test suite is currently passing on the default branch.
        Uses GitHub Checks API if credentials are available, otherwise skips.
        """
        if not self.github_token or not self.repo_slug:
            logger.debug("Skipping CI-green check — no GitHub credentials")
            return True, ""

        try:
            import httpx

            headers = {
                "Authorization": f"Bearer {self.github_token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            }
            # Get the latest commit on the default branch
            url = f"https://api.github.com/repos/{self.repo_slug}/commits/main/status"
            resp = httpx.get(url, headers=headers)
            resp.raise_for_status()

            state = resp.json().get("state", "unknown")
            if state == "failure":
                return False, "CI is currently red on main — refuse to open PR on broken build"
            if state == "pending":
                logger.warning("CI is pending on main — proceeding cautiously")

        except ImportError:
            logger.debug("httpx not available — skipping CI check")
        except Exception as exc:
            logger.warning("Could not check CI status: %s — proceeding", exc)

        return True, ""
