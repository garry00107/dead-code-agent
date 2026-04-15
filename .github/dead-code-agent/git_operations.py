"""
Git Operations — branch management, file mutations, commit, push, PR creation.

Uses subprocess for git commands and httpx for GitHub API.
Designed for CI environments where git is pre-configured with bot credentials.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import sys as _sys
from pathlib import Path as _Path
_SCRIPT_DIR = _Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in _sys.path:
    _sys.path.insert(0, str(_SCRIPT_DIR))

from models import ZombieCluster

logger = logging.getLogger(__name__)


class GitOperations:
    """
    Handles all git and GitHub API interactions for PR generation.
    """

    def __init__(
        self,
        repo_path: Path,
        github_token: Optional[str] = None,
        repo_slug: Optional[str] = None,
    ):
        self.repo_path = repo_path
        self.github_token = github_token or os.environ.get("GITHUB_TOKEN", "")
        self.repo_slug = repo_slug  # "owner/repo"

    # ─── Branch Management ──────────────────────────────────────────

    def create_branch(self, cluster_id: str) -> str:
        """Create a new branch for this cluster's PR."""
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
        short = cluster_id[:8]
        name = f"dead-code/cluster-{short}-{ts}"

        # Ensure we're on the latest main
        self._run_git("checkout", "main")
        self._run_git("pull", "--ff-only", "origin", "main")
        self._run_git("checkout", "-b", name)

        logger.info("Created branch: %s", name)
        return name

    def abort_branch(self, branch_name: str) -> None:
        """Abandon a branch — reset and delete it."""
        try:
            self._run_git("checkout", "main", check=False)
            self._run_git("reset", "--hard", "HEAD", check=False)
            self._run_git("branch", "-D", branch_name, check=False)
            logger.info("Aborted branch: %s", branch_name)
        except Exception as exc:
            logger.warning("Branch abort cleanup failed: %s", exc)

    # ─── File Mutations ─────────────────────────────────────────────

    def apply_deletion_plan(self, steps: list[dict]) -> None:
        """Execute each step in the deletion plan."""
        for i, step in enumerate(steps):
            action = step["action"]
            file_path = step["file"]
            logger.info(
                "Step %d/%d: %s on %s", i + 1, len(steps), action, file_path,
            )

            if action == "delete_file":
                self._delete_file(file_path)
            elif action == "remove_lines":
                self._remove_lines(file_path, step["start_line"], step["end_line"])
            elif action == "cleanup_imports":
                self._cleanup_imports(file_path)
            else:
                logger.warning("Unknown action: %s — skipping", action)

    def _delete_file(self, file_path: str) -> None:
        """Delete a file from the working tree."""
        full_path = self.repo_path / file_path
        if full_path.exists():
            full_path.unlink()
            logger.info("Deleted: %s", file_path)
        else:
            logger.warning("File already missing: %s", file_path)

    def _remove_lines(self, file_path: str, start: int, end: int) -> None:
        """
        Remove lines [start, end] (1-indexed, inclusive) from a file.
        Cleans up trailing blank lines left by the deletion.
        """
        full_path = self.repo_path / file_path
        lines = full_path.read_text(encoding="utf-8").splitlines(keepends=True)

        # Lines are 1-indexed; keep everything outside [start, end]
        kept = lines[: start - 1] + lines[end:]

        # Remove trailing blank lines left by deletion (but keep at least one newline)
        while len(kept) > 1 and kept[-1].strip() == "" and kept[-2].strip() == "":
            kept.pop()

        # Ensure file ends with exactly one newline
        if kept and not kept[-1].endswith("\n"):
            kept[-1] += "\n"
        elif not kept:
            kept = ["\n"]

        full_path.write_text("".join(kept), encoding="utf-8")
        logger.info("Removed lines %d–%d from %s", start, end, file_path)

    def _cleanup_imports(self, file_path: str) -> None:
        """
        Remove orphaned imports from a file.
        - Python: uses autoflake if available
        - TypeScript/JavaScript: basic regex cleanup
        """
        full_path = self.repo_path / file_path

        if file_path.endswith(".py"):
            self._cleanup_python_imports(full_path)
        elif file_path.endswith((".ts", ".tsx", ".js", ".jsx")):
            self._cleanup_ts_imports(full_path)

    def _cleanup_python_imports(self, path: Path) -> None:
        """Run autoflake to remove unused imports. No-op if autoflake isn't installed."""
        try:
            result = subprocess.run(
                [
                    "python", "-m", "autoflake",
                    "--in-place",
                    "--remove-all-unused-imports",
                    "--remove-unused-variables",
                    str(path),
                ],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                logger.info("Cleaned imports in %s via autoflake", path.name)
            else:
                logger.debug("autoflake returned %d: %s", result.returncode, result.stderr)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            logger.debug("autoflake not available — skipping import cleanup for %s", path.name)

    def _cleanup_ts_imports(self, path: Path) -> None:
        """
        Basic TypeScript/JavaScript import cleanup.
        Removes imports where none of the imported names appear in the rest of the file.
        """
        import re

        try:
            source = path.read_text(encoding="utf-8")
            lines = source.splitlines(keepends=True)
            cleaned: list[str] = []

            for line in lines:
                # Match: import { X, Y } from '...'
                match = re.match(
                    r"^import\s+\{([^}]+)\}\s+from\s+['\"].*['\"];?\s*$", line,
                )
                if match:
                    names = [n.strip().split(" as ")[-1].strip() for n in match.group(1).split(",")]
                    rest_of_file = source.replace(line, "")
                    # Keep import if ANY imported name is still used
                    if any(re.search(r"\b" + re.escape(name) + r"\b", rest_of_file) for name in names if name):
                        cleaned.append(line)
                    else:
                        logger.info("Removed unused import: %s", line.strip())
                        continue
                else:
                    cleaned.append(line)

            path.write_text("".join(cleaned), encoding="utf-8")
        except Exception as exc:
            logger.debug("TS import cleanup failed for %s: %s", path.name, exc)

    # ─── Commit & Push ──────────────────────────────────────────────

    def commit_all(self, cluster: ZombieCluster, branch_name: str) -> str:
        """Stage all changes, commit with structured message, push, return SHA."""
        self._run_git("add", "-A")

        symbols_str = ", ".join(s.fqn for s in cluster.symbols[:10])
        if len(cluster.symbols) > 10:
            symbols_str += f" +{len(cluster.symbols) - 10} more"

        msg = (
            f"chore(dead-code): remove cluster {cluster.cluster_id[:8]}\n\n"
            f"Auto-generated by Dead Code Removal Agent\n"
            f"Confidence: {cluster.combined_confidence:.0%}\n"
            f"Symbols: {symbols_str}"
        )
        self._run_git("commit", "-m", msg)
        self._run_git("push", "origin", branch_name)

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
        )
        sha = result.stdout.strip()
        logger.info("Committed and pushed %s (SHA: %s)", branch_name, sha[:8])
        return sha

    # ─── Pull Request ───────────────────────────────────────────────

    def open_pull_request(
        self,
        branch: str,
        pr_meta: dict,
        base_branch: str = "main",
        labels: list[str] | None = None,
    ) -> str:
        """
        Open a PR via GitHub API. Returns the PR URL.
        Uses httpx for the API call — no PyGithub dependency.
        """
        import httpx

        labels = (labels or []) + ["dead-code", "auto-generated"]

        headers = {
            "Authorization": f"Bearer {self.github_token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

        # Create PR
        create_url = f"https://api.github.com/repos/{self.repo_slug}/pulls"
        payload = {
            "title": pr_meta["title"],
            "body": pr_meta["body"],
            "head": branch,
            "base": base_branch,
            "draft": False,
        }

        resp = httpx.post(create_url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        pr_data = resp.json()
        pr_url = pr_data["html_url"]
        pr_number = pr_data["number"]

        logger.info("Opened PR #%d: %s", pr_number, pr_url)

        # Add labels (separate API call — create_pull doesn't accept labels)
        if labels:
            try:
                labels_url = f"https://api.github.com/repos/{self.repo_slug}/issues/{pr_number}/labels"
                httpx.post(labels_url, headers=headers, json={"labels": labels}, timeout=15)
                logger.info("Added labels: %s", labels)
            except Exception as exc:
                logger.warning("Failed to add labels to PR #%d: %s", pr_number, exc)

        return pr_url

    # ─── Helpers ────────────────────────────────────────────────────

    def _run_git(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        """Run a git command in the repo directory."""
        cmd = ["git"] + list(args)
        logger.debug("Running: %s", " ".join(cmd))
        return subprocess.run(
            cmd,
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            check=check,
            timeout=60,
        )
